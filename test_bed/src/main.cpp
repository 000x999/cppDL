#include "neural_core/neural_network.hpp"
#include "tokenizer_core/tokenizer.hpp"
#include "logger_core/dual_output.hpp"
#include "tensor_core/tensor.hpp"
#include "attention_core/attention.hpp"
#include "model_core/gpt2.hpp"
#include "safetensor_core/safetensor_reader.h"
#include <stdlib.h>
#include <chrono>
#include <fstream>
#include <cfloat>
#include <streambuf>
#include <chrono>
#include <thread>
#include <x86intrin.h>
#include <limits>

size_t estimate_flops_per_token(gpt2::config& cfg, size_t seq_len) {
  size_t embed_dim = cfg.embed_dim;
  size_t num_heads = cfg.num_heads;
  size_t num_layers = cfg.num_layers;
  size_t vocab_size = cfg.vocab_size;
  size_t head_dim = embed_dim / num_heads;
  size_t ffn_hidden = embed_dim * 4;
  
  size_t flops = 0;
  
  for (size_t l = 0; l < num_layers; l++) {
    flops += 5 * seq_len * embed_dim;
    flops += 3 * 2 * seq_len * embed_dim * embed_dim;
    flops += 2 * num_heads * seq_len * seq_len * head_dim;
    flops += 5 * num_heads * seq_len * seq_len;
    flops += 2 * num_heads * seq_len * seq_len * head_dim;
    flops += 2 * seq_len * embed_dim * embed_dim;
    flops += 5 * seq_len * embed_dim;
    flops += 2 * seq_len * embed_dim * ffn_hidden;
    flops += 10 * seq_len * ffn_hidden;
    flops += 2 * seq_len * ffn_hidden * embed_dim;
  }
  
  flops += 5 * seq_len * embed_dim;
  
  flops += 2 * seq_len * embed_dim * vocab_size;
  
  return flops;
}


void save_ppm(const std::string& name, const float* data, int width, int height) {
if (!data) return;
  std::string filename = name + ".ppm";
  std::ofstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error opening file for visualization: " << filename << "\n";
    return;
  }
  
  f << "P3\n" << width << " " << height << "\n255\n";
  
  float min_v = 1e9, max_v = -1e9;
  for(int i=0; i< width * height; ++i) {
      if(data[i] < min_v) min_v = data[i];
      if(data[i] > max_v) max_v = data[i];
  }
  
  for(int i=0; i< width * height; ++i) {
    float t = (data[i] - min_v) / (max_v - min_v + 1e-8f);
      
    // for black = high, white = low  
    //int intensity = (int)((1.0f - t) * 255.0f);
    int intensity = (int)(t * 255.0f);
    
    f << intensity << " " << intensity << " " << intensity << " ";
    
    if((i+1) % width == 0) f << "\n";
  }
  f.close();
  std::cout << "Saved grayscale visualization: " << filename << "\n";
}

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void tokenizer_test(){
  bpe::bpe_tokenizer tokenizer;
  std::string file_path = "core/include/tokenizer_core/token_models/data_set.txt"; 
  std::ifstream in_file {file_path};
  std::string training_text {std::istreambuf_iterator<char>(in_file), std::istreambuf_iterator<char>()};
  if(!in_file){std::cout << "FNF" << '\n';}
  size_t num_merges = 10000;
  std::cout << "Training BPE tokenizer with " << num_merges << " merges...\n";
  tokenizer.train(training_text, num_merges);
  std::string testText = "I am testing out a large training data set for the tokenizer, we will see if this works properly.";
  std::vector<bpe::g_token_id> encoded_ids = tokenizer.encode(testText);
  std::cout << "Encoded IDs for test text:\n";
  int id_count = 0; 
  for (const auto& id : encoded_ids) {
    std::cout << "Encoded ID: " << id << " -> '" << tokenizer.decode({id}) << "'\n";
    id_count++;
  }
  std::string decoded_text = tokenizer.decode(encoded_ids);
 // std::cout << "Decoded text: " << decoded_text << std::endl;
  if (decoded_text == testText) {
    std::cout << "***NOTE***: Encoding/decoding is lossless" << std::endl;
  } 
  else {
    std::cout << "***WARNING***: Encoding/decoding is not lossless" << std::endl;
  }
  tokenizer.save_model("core/include/tokenizer_core/token_models/vocab.txt", "core/include/tokenizer_core/token_models/bpe_merges.txt");
  std::cout << "Model saved to files" << std::endl;
  tokenizer.print_model_stats();
}

void inference_test(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  size_t layer_1_input  = 8192 * 2;
  size_t layer_1_output = 4096 * 2;
  size_t layer_2_input  = 4096 * 2; 
  size_t layer_2_output = 2048 * 2;
  size_t layer_3_input  = 2048 * 2; 
  size_t layer_3_output = 1024 * 2;
  size_t layer_4_input  = 1024 * 2;
  size_t layer_4_output = 512  * 2;
  size_t batch_size     = 32   * 2;
  
  neural::nn inf;
  inf.add_linear(layer_1_input, layer_1_output);
  inf.add_relu(); 
  inf.add_linear(layer_2_input, layer_2_output);
  inf.add_relu();
  inf.add_linear(layer_3_input, layer_3_output);
  inf.add_relu(); 
  inf.add_linear(layer_4_input, layer_4_output);
  inf.add_sigmoid();

  size_t arena_size = inf.mem_reqs(); 
  
  neural::alloc_pool persistent_arena(arena_size * sizeof(float)); 
  inf.init(persistent_arena);
  
  neural::alloc_pool temp_arena(arena_size * sizeof(float)); 
  float *input_data = temp_arena.arena.nn_alloc<float>(batch_size * layer_1_input);

  for(size_t i = 0; i < layer_1_input * batch_size; ++i){
    input_data[i] = dist(gen); 
  }
  
  std::cout << "network shape: " << '\n' << "first layer: " << layer_1_input << " x " << layer_1_output << '\n'; 
  neural::neural_view input_tensor; 
  input_tensor.tensor.tensor_data      = input_data; 
  input_tensor.tensor.shape.ndim       = 2; 
  input_tensor.tensor.shape.dims[0]    = batch_size; 
  input_tensor.tensor.shape.dims[1]    = layer_1_input; 
  input_tensor.tensor.shape.strides[0] = layer_1_input;
  input_tensor.tensor.shape.strides[1] = 1;

  save_ppm("inputdata", input_tensor.tensor.tensor_data, input_tensor.tensor.shape.dims[0], input_tensor.tensor.shape.dims[1]); 
  save_ppm("allweights", inf.save_weights(), input_tensor.tensor.shape.dims[0], input_tensor.tensor.shape.dims[1]);
  auto start = nanos();
    neural::neural_view res_tensor = inf.forward(input_tensor, temp_arena);
  auto end   = nanos();
  std::cout << "forward time: " << (end - start) * 1e-6 << "ms\n"; 
  save_ppm("infweights", res_tensor.tensor.tensor_data, res_tensor.tensor.shape.dims[0], res_tensor.tensor.shape.dims[1]); 

  std::cout << "output shape [" << res_tensor.tensor.shape.dims[0] 
            << "," << res_tensor.tensor.shape.dims[1] << "]\n"; 

  std::cout << "output values: "; 
  for(size_t i = 0; i < res_tensor.tensor.shape.dims[1]; ++i){
    std::cout << res_tensor.tensor.tensor_data[i] << " "; 
  }
  std::cout << '\n'; 
}

void attention_test(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f); 
  
  size_t embed_dim = 32;
  size_t num_heads = 1;
  size_t head_dim  = embed_dim / num_heads; 
  size_t seq_len   = 4;  
  
  atten::attention attn(embed_dim, num_heads);
  
  size_t weight_size = embed_dim * head_dim;
  size_t total_weights = weight_size * 4;    
  atten::atten_pool persistent_arena(total_weights * sizeof(float) * 50); 
  attn.init(persistent_arena); 
  
  atten::atten_pool temp_arena(seq_len * embed_dim * 200 * sizeof(float)); 
  
  float *input_data = temp_arena.arena.nn_alloc<float>(seq_len * embed_dim); 
  for(size_t i = 0; i < seq_len * embed_dim; ++i){
      input_data[i] = dist(gen); 
  }
  
  float *wq_data = temp_arena.arena.nn_alloc<float>(weight_size); 
  float *wk_data = temp_arena.arena.nn_alloc<float>(weight_size);
  float *wv_data = temp_arena.arena.nn_alloc<float>(weight_size);
  float *wo_data = temp_arena.arena.nn_alloc<float>(weight_size);
  
  for(size_t i = 0; i < weight_size; ++i){
    wq_data[i] = dist(gen); 
    wk_data[i] = dist(gen); 
    wv_data[i] = dist(gen); 
    wo_data[i] = dist(gen); 
  }
  
  tens::tensor input_tensor; 
  input_tensor.tensor_data      = input_data;
  input_tensor.shape.ndim       = 2; 
  input_tensor.shape.dims[0]    = seq_len;  
  input_tensor.shape.dims[1]    = embed_dim;  
  input_tensor.shape.strides[0] = embed_dim; 
  input_tensor.shape.strides[1] = 1;
   
  //save_ppm("inputdata", input_tensor.tensor_data, input_tensor.shape.dims[0], input_tensor.shape.dims[1]); 
  attn.load_weights(wq_data, wk_data, wv_data, wo_data);
  auto start = nanos();  
  auto output_tensor = attn.forward(input_tensor, temp_arena); 
  auto end   = nanos();
  std::cout << "forward time: " << (end - start) * 1e-6 << "ms\n";
  
  std::cout << "Output shape: [" << output_tensor.shape.dims[0] 
            << ", " << output_tensor.shape.dims[1] << "]" << std::endl;
  save_ppm("attenweights", output_tensor.tensor_data, output_tensor.shape.dims[0], output_tensor.shape.dims[1]);
}

void multi_head_attention_test(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f); 

  size_t embed_dim = 8192;
  size_t num_heads = 8;
  size_t head_dim  = embed_dim / num_heads; 
  size_t seq_len   = 512;  

  std::cout << "=== Multi-Head Attention Test ===" << std::endl;
  std::cout << "embed_dim: " << embed_dim << std::endl;
  std::cout << "num_heads: " << num_heads << std::endl;
  std::cout << "head_dim: " << head_dim << std::endl;
  std::cout << "seq_len: " << seq_len << std::endl;

  atten::multi_head_attention attn(embed_dim, num_heads);

  size_t weight_size = embed_dim * embed_dim;
  size_t total_weights = weight_size * 4;    
  atten::atten_pool persistent_arena(total_weights * sizeof(float) + 4096); 
  attn.init(persistent_arena); 

  atten::atten_pool temp_arena(seq_len * embed_dim * 200 * sizeof(float)); 

  float *input_data = temp_arena.arena.nn_alloc<float>(seq_len * embed_dim); 
  for(size_t i = 0; i < seq_len * embed_dim; ++i){
    input_data[i] = dist(gen); 
  }

  float *wq_data = temp_arena.arena.nn_alloc<float>(weight_size); 
  float *wk_data = temp_arena.arena.nn_alloc<float>(weight_size);
  float *wv_data = temp_arena.arena.nn_alloc<float>(weight_size);
  float *wo_data = temp_arena.arena.nn_alloc<float>(weight_size);

  for(size_t i = 0; i < weight_size; ++i){
    wq_data[i] = dist(gen); 
    wk_data[i] = dist(gen); 
    wv_data[i] = dist(gen); 
    wo_data[i] = dist(gen); 
  }

  tens::tensor input_tensor; 
  input_tensor.tensor_data      = input_data;
  input_tensor.shape.ndim       = 2; 
  input_tensor.shape.dims[0]    = seq_len;  
  input_tensor.shape.dims[1]    = embed_dim;  
  input_tensor.shape.strides[0] = embed_dim; 
  input_tensor.shape.strides[1] = 1;

  //attn.load_weights(wq_data, wk_data, wv_data, wo_data);

  auto start = nanos();  
  auto output_tensor = attn.forward(input_tensor, temp_arena); 
  auto end = nanos();

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "Forward time: " << (end - start) * 1e-6 << "ms" << std::endl;

  std::cout << "\n=== Shape Check ===" << std::endl;
  std::cout << "Input shape:  [" << input_tensor.shape.dims[0] << ", " << input_tensor.shape.dims[1] << "]" << std::endl;
  std::cout << "Output shape: [" << output_tensor.shape.dims[0] << ", " << output_tensor.shape.dims[1] << "]" << std::endl;
  
  bool shape_match = (input_tensor.shape.dims[0] == output_tensor.shape.dims[0]) && 
                     (input_tensor.shape.dims[1] == output_tensor.shape.dims[1]);
  std::cout << "Shapes match: " << (shape_match ? "YES" : "NO (BAD!)") << std::endl;

  std::cout << "\n=== Output Statistics ===" << std::endl;
  float min_val = FLT_MAX, max_val = -FLT_MAX;
  float sum = 0.0f;
  size_t count = output_tensor.shape.dims[0] * output_tensor.shape.dims[1];
  bool has_nan = false;
  bool has_inf = false;

  for (size_t i = 0; i < count; ++i) {
    float v = output_tensor.tensor_data[i];
    if (std::isnan(v)) has_nan = true;
    if (std::isinf(v)) has_inf = true;
    if (v < min_val) min_val = v;
    if (v > max_val) max_val = v;
    sum += v;
  }

  std::cout << "Min:  " << min_val << std::endl;
  std::cout << "Max:  " << max_val << std::endl;
  std::cout << "Mean: " << sum / count << std::endl;
  std::cout << "Has NaN: " << (has_nan ? "YES (BAD!)" : "No") << std::endl;
  std::cout << "Has Inf: " << (has_inf ? "YES (BAD!)" : "No") << std::endl;

  std::cout << "\n=== Sample Output Values ===" << std::endl;
  for (size_t i = 0; i < 10 && i < count; ++i) {
      std::cout << output_tensor.tensor_data[i] << " ";
  }
  std::cout << std::endl;

  save_ppm("attenweights", output_tensor.tensor_data, output_tensor.shape.dims[0], output_tensor.shape.dims[1]);
  
  std::cout << "\n=== Test Complete ===" << std::endl;
}

void gemm_test(float A){
  std::cout << "=== AVX512 GEMM TEST ===" <<'\n'; 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f); 
  atten::atten_pool temp_arena(3 * (A * A) * sizeof(float) + 4096);

  float *data_ptr_a = temp_arena.arena.nn_alloc<float>(A * A); 
  float *data_ptr_b = temp_arena.arena.nn_alloc<float>(A * A);
  float *data_ptr_c = temp_arena.arena.nn_alloc<float>(A * A);

  for(size_t i = 0; i < A * A; ++i){
    data_ptr_a[i] = dist(gen);
    data_ptr_b[i] = dist(gen); 
  }

  level3::mat_ops_view mat_a {
    .row_view = (size_t)A, 
    .col_view = (size_t)A, 
    .leading_dimension = (size_t)A,
    .data_view = data_ptr_a
  };
  
  level3::mat_ops_view mat_b {
    .row_view = (size_t)A, 
    .col_view = (size_t)A, 
    .leading_dimension = (size_t)A,
    .data_view = data_ptr_b
  };

  level3::mat_ops_view C {
    .row_view = (size_t)A, 
    .col_view = (size_t)A, 
    .leading_dimension = (size_t)A, 
    .data_view = data_ptr_c 
  };

  double totalOps = 2.0 * double(A) * double(A) * double(A);
  double gflopFactor = 1.0e-9;
  std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl; 

  auto start = nanos(); 
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose,level3::transpose_gemm::no_transpose, mat_a, mat_b, 1.0f, 0.0f, C);
  auto end = nanos();
  
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX512 MatMul: " << optTime
            << "s, GFLOP/S = " << optGflops << "\n";
}

int main(int argc, char* argv[]){
  const char* model_path = "model.safetensors";
  
  if (argc > 1) {
      model_path = argv[1];
  }
  
  size_t model_arena_size = 1024ULL * 1024ULL * 1024ULL; 
  size_t temp_arena_size = 1024ULL * 1024ULL * 512ULL;   
  
  memory::neural_arena model_arena(model_arena_size);
  tens::tensor_pool temp_pool(temp_arena_size);
  
  gpt2::model model;
  gpt2::init_model(&model);
 
  gpt2::tokenizer tokenizer;
  if (!tokenizer.load("vocab.json")) {
    std::printf("Warning: Could not load vocab.json. Output will be IDs only.\n");
  }

  if (!gpt2::load_model(&model, model_path, model_arena)) {
    std::printf("Failed to load model from: %s\n", model_path);
    return 1;
  }
  
  size_t max_seq_len = 1024;
  float* sequence = (float*)std::malloc(max_seq_len * sizeof(float));
  
  sequence[0] = 15496.0f; 
  size_t seq_len = 1;
  
  std::printf("\n");
  std::printf("Model config:\n");
  std::printf("  vocab_size:  %zu\n", model.cfg.vocab_size);
  std::printf("  embed_dim:   %zu\n", model.cfg.embed_dim);
  std::printf("  num_layers:  %zu\n", model.cfg.num_layers);
  std::printf("  num_heads:   %zu\n", model.cfg.num_heads);
  std::printf("  max_seq_len: %zu\n", model.cfg.max_seq_len);
  std::printf("\n");
  
  std::printf("Starting generation...\n");
  std::printf("Input token: %d\n\n", (int)sequence[0]);
  
  int max_new_tokens = 50;
  
  uint64_t total_time_ns = 0;
  size_t total_flops = 0;
  int tokens_generated = 0;
  
  std::printf("Generated token IDs: ");

  std::printf("Generated text:\n");
    
  std::printf("%s", tokenizer.decode((int)sequence[0]).c_str());
  std::fflush(stdout);
  
  for (int i = 0; i < max_new_tokens; i++) {
    if (i < 2) { 
      std::printf("\n[DEBUG] Forward pass %d, seq_len=%zu, tokens: ", i, seq_len);
      for (size_t t = 0; t < seq_len && t < 5; t++) std::printf("%.0f ", sequence[t]);
      std::printf("\n");
    }

    tens::tensor input;
    std::memset(&input, 0, sizeof(tens::tensor));
    input.shape.ndim = 1;
    input.shape.dims[0] = seq_len;
    input.shape.strides[0] = 1;
    input.tensor_data = sequence;
    
    uint64_t start = nanos();
    tens::tensor logits = gpt2::forward(&model, input, temp_pool);
    uint64_t end = nanos();
    
    uint64_t elapsed_ns = end - start;
    size_t flops = estimate_flops_per_token(model.cfg, seq_len);
    
    total_time_ns += elapsed_ns;
    total_flops += flops;
    tokens_generated++;
    
    float* last_token_logits = logits.tensor_data + (seq_len - 1) * model.cfg.vocab_size;
    int next_token = gpt2::sample_top_k_avx512(last_token_logits, model.cfg.vocab_size, 40, 0.75f, temp_pool );
   
    std::string token_str = tokenizer.decode(next_token);
    std::printf("%s", token_str.c_str());
    std::fflush(stdout);
    std::printf("%d ", next_token);
    std::fflush(stdout);
    
    if (seq_len < max_seq_len) {
      sequence[seq_len] = (float)next_token;
      seq_len++;
    } else {
      std::printf("\n[Max sequence length reached]\n");
      break;
    }
    
    if (next_token == 50256) {
      std::printf("\n[EOS]\n");
      break;
    }
    
    temp_pool.arena.nn_reset();
      
    for (size_t layer = 0; layer < model.cfg.num_layers; layer++) {
      if (model.atten_pools[layer]) {
        model.atten_pools[layer]->arena.nn_reset();
      }
    }
}
  
  std::printf("\n\n");
  
  std::printf("Full sequence (%zu tokens): ", seq_len);
  for (size_t i = 0; i < seq_len; i++) {
    std::printf("%d ", (int)sequence[i]);
  }
  std::printf("\n\n");
  
  double total_time_s = (double)total_time_ns / 1e9;
  double total_gflops = (double)total_flops / 1e9;
  double gflops_per_sec = total_gflops / total_time_s;
  double tokens_per_sec = (double)tokens_generated / total_time_s;
  double ms_per_token = (total_time_s * 1000.0) / (double)tokens_generated;
  
  std::printf("=== Performance Summary ===\n");
  std::printf("Tokens generated:  %d\n", tokens_generated);
  std::printf("Total time:        %.3f s\n", total_time_s);
  std::printf("Tokens/sec:        %.2f\n", tokens_per_sec);
  std::printf("ms/token:          %.2f\n", ms_per_token);
  std::printf("Total GFLOP:       %.2f\n", total_gflops);
  std::printf("GFLOP/s:           %.2f\n", gflops_per_sec);
  std::printf("===========================\n");
  
  std::free(sequence);
  gpt2::free_model(&model);
  
  return 0;
}
