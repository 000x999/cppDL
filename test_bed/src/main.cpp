#include "neural_core/neural_network.hpp"
#include "tokenizer_core/tokenizer.hpp"
#include "logger_core/dual_output.hpp"
#include "tensor_core/tensor.hpp"
#include "attention_core/attention.hpp"
#include <stdlib.h>
#include <chrono>
#include <fstream>
#include <cfloat>
#include <streambuf>
#include <chrono>
#include <thread>
#include <x86intrin.h>
#include <limits> 

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
      
    int r, g, b;
      
    if (t < 0.5f) {
      float local_t = t * 2.0f;
      r = 0;
      g = (int)(local_t * 255.0f);
      b = (int)((1.0f - local_t) * 255.0f);
    }else{
      float local_t = (t - 0.5f) * 2.0f;
      r = (int)(local_t * 255.0f);
      g = (int)((1.0f - local_t) * 255.0f);
      b = 0;
    }
      
    f << r << " " << g << " " << b << " ";
    if((i+1) % width == 0) f << "\n";
  }
  f.close();
  std::cout << "Saved colored visualization: " << filename << "\n";
}

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void tokenizer_test(){
  bpe::bpe_tokenizer tokenizer;
  std::string file_path = "core/include/tokenizer_core/token_models/DataSet.txt"; 
  std::ifstream in_file {file_path};
  std::string training_text {std::istreambuf_iterator<char>(in_file), std::istreambuf_iterator<char>()};
  if(!in_file){std::cout << "FNF" << '\n';}
  size_t num_merges = 100;
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
  std::cout << "Decoded text: " << decoded_text << std::endl;
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

  size_t layer_1_input  = 8192;
  size_t layer_1_output = 4096;
  size_t layer_2_input  = 4096; 
  size_t layer_2_output = 2048;
  size_t layer_3_input  = 2048; 
  size_t layer_3_output = 1024;
  size_t layer_4_input  = 1024;
  size_t layer_4_output = 512;
  size_t batch_size     = 32;
  
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
    
    size_t embed_dim = 8192;
    size_t num_heads = 1;
    size_t head_dim  = embed_dim / num_heads; 
    size_t seq_len   = 512;  
    
    atten::attention attn(embed_dim, num_heads);
    
    size_t weight_size = embed_dim * head_dim;
    size_t total_weights = weight_size * 4;    
    atten::atten_pool persistent_arena(total_weights * sizeof(float)); 
    attn.init(persistent_arena); 
    
    atten::atten_pool temp_arena(seq_len * embed_dim * 100 * sizeof(float)); 
    
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

int main(){
  //inference_test(); 
  attention_test(); 
}
