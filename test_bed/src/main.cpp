#include "../../core/include/functions_core/functions.hpp"
#include "../../core/include/neural_core/neural_network.hpp"
#include "../../core/include/tokenizer_core/tokenizer.hpp"
#include "../../core/include/logger_core/dual_output.hpp"
#include "../../core/utils/dashboard.hpp"
#include <stdlib.h>
#include <chrono>
#include <fstream>
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
    
    // Find Min/Max for normalization
    float min_v = 1e9, max_v = -1e9;
    for(int i=0; i<width*height; ++i) {
        if(data[i] < min_v) min_v = data[i];
        if(data[i] > max_v) max_v = data[i];
    }
    
    // Write pixels
    for(int i=0; i<width*height; ++i) {
        // Normalize to 0..1
        float t = (data[i] - min_v) / (max_v - min_v + 1e-8f);
        
        // Heatmap Color Map (Blue -> Green -> Red)
        int r, g, b;
        
        if (t < 0.5f) {
            // Blue (0,0,255) -> Green (0,255,0)
            float local_t = t * 2.0f;
            r = 0;
            g = (int)(local_t * 255.0f);
            b = (int)((1.0f - local_t) * 255.0f);
        } else {
            // Green (0,255,0) -> Red (255,0,0)
            float local_t = (t - 0.5f) * 2.0f;
            r = (int)(local_t * 255.0f);
            g = (int)((1.0f - local_t) * 255.0f);
            b = 0;
        }
        
        f << r << " " << g << " " << b << " ";
        if((i+1)%width == 0) f << "\n";
    }
    f.close();
    std::cout << "Saved colored visualization: " << filename << "\n";
}



uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void neural_network_test(){ 
  init_console_utf8();
  ansi_hide_cursor();
  ansi_clear_screen();

  size_t epochmax   = 1000;
  size_t squeezemax = 0; 
  auto start = nanos(); 

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-50.0f, 100.0f);

  std::vector<float> inputVals;
  inputVals.reserve(10000);
  for(size_t i = 0; i < 10000; ++i) {
    inputVals.emplace_back(dist(gen));
  }

  std::vector<float> targetVals = {1.0f};
  float eta = 0.00001f; 

  neural::nn net;
  net.add_linear (inputVals.size(),10000); 
  net.add_relu   (10000);
  net.add_linear (10000,1000);
  net.add_sigmoid(1000);
  net.add_linear (1000,1);
  net.add_loss   (std::make_unique<neural::mse_loss>());

  std::vector<float> out; 
  float loss = 0.0f; 

  std::vector<float> loss_hist;
  loss_hist.reserve(epochmax);

  const size_t input_dim   = inputVals.size();
  const size_t hidden1_dim = 10000;
  const size_t hidden2_dim = 1000;
  const size_t output_dim  = 1;

  const int term_width = 120;  

  const int train_w    = 78;
  const int train_h    = 13;
  const int net_w      = 75;
  const int net_h      = 9;
  const int loss_w     = term_width - 4;
  const int loss_h     = 7;

  const int train_top  = 2;
  const int train_left = 2;
  const int net_top    = 2;
  const int net_left   = train_left + train_w + 2;
  const int loss_top   = train_top + train_h + 1;
  const int loss_left  = 2;

  ansi_move(1, 2);
  fg256(45); ansi_bold();
  std::cout << " cppDL single-sample training ";
  ansi_reset();

  draw_box(train_top, train_left, train_w, train_h, "Training");
  draw_box(net_top,   net_left,   net_w,   net_h,   "Network");
  draw_box(loss_top,  loss_left,  loss_w,  loss_h,  "Loss history");

  ansi_move(net_top + 2, net_left + 3);
  fg256(15); ansi_bold(); std::cout << "Input dim "; ansi_reset();
  std::cout << ": " << input_dim << "          ";

  ansi_move(net_top + 3, net_left + 3);
  fg256(15); ansi_bold(); std::cout << "Hidden1   "; ansi_reset();
  std::cout << ": " << hidden1_dim << "         ";

  ansi_move(net_top + 4, net_left + 3);
  fg256(15); ansi_bold(); std::cout << "Hidden2   "; ansi_reset();
  std::cout << ": " << hidden2_dim << "         ";

  ansi_move(net_top + 5, net_left + 3);
  fg256(15); ansi_bold(); std::cout << "Output dim"; ansi_reset();
  std::cout << ": " << output_dim << "          ";

  ansi_move(net_top + 7, net_left + 3);
  ansi_dim();
  std::cout << "Architecture: Linear -> ReLU -> Linear -> Sigmoid -> Linear";
  ansi_reset();

  ansi_move(loss_top + 1, loss_left + 3);
  fg256(15); ansi_bold(); std::cout << "Last losses"; ansi_reset();

  for(size_t epoch = 0; epoch < epochmax; ++epoch){
    out            = net.forward(inputVals); 
    loss           = net.get_loss(targetVals);
    auto derivOut  = net.get_grad(targetVals);
    net.backwards(derivOut); 
    net.update(eta);

    loss_hist.push_back(loss);
    if (loss_hist.size() > 200) {
      loss_hist.erase(loss_hist.begin());  
    }

    double elapsed_s = (nanos() - start) * 1e-9;
    double prog      = double(epoch + 1) / double(epochmax);

    ansi_move(train_top + 2, train_left + 3);
    fg256(15); ansi_bold(); std::cout << "Epoch"; ansi_reset();
    std::cout << ": " << std::setw(5) << epoch << " / " << std::setw(5) << (epochmax - 1) << "       ";

    ansi_move(train_top + 3, train_left + 3);
    fg256(15); ansi_bold(); std::cout << "Time "; ansi_reset();
    std::cout << ": ";
    fg256(220);
    std::cout << std::fixed << std::setprecision(2)
              << std::setw(8) << elapsed_s << " s   ";
    ansi_reset();

    ansi_move(train_top + 4, train_left + 3);
    fg256(15); ansi_bold(); std::cout << "Loss "; ansi_reset();
    std::cout << ": ";
    fg256(207);
    std::cout << std::fixed << std::setprecision(6)
              << std::setw(12) << loss << "   ";
    ansi_reset();

    ansi_move(train_top + 5, train_left + 3);
    fg256(15); ansi_bold(); std::cout << "Output"; ansi_reset();
    std::cout << ": ";
    fg256(82);
    std::cout << std::fixed << std::setprecision(6)
              << std::setw(12) << (out.empty() ? 0.0f : out[0]) << "   ";
    ansi_reset();

    ansi_move(train_top + 6, train_left + 3);
    fg256(15); ansi_bold(); std::cout << "Target"; ansi_reset();
    std::cout << ": ";
    fg256(82);
    std::cout << std::fixed << std::setprecision(6)
              << std::setw(12) << targetVals[0] << "   ";
    ansi_reset();

    ansi_move(train_top + 7, train_left + 3);
    fg256(15); ansi_bold(); std::cout << "LR   "; ansi_reset();
    std::cout << ": ";
    fg256(33);
    std::cout << std::fixed << std::setprecision(8)
              << std::setw(14) << eta << "   ";
    ansi_reset();

    ansi_move(train_top + 8, train_left + 3);
    ansi_dim(); std::cout << "Epoch progress:"; ansi_reset();
    draw_bar(train_top + 8, train_left + 20, train_w - 24, prog);

    draw_loss_sparkline(loss_top + 3, loss_left + 3, loss_w - 6, loss_hist);

    ansi_move(loss_top + 5, loss_left + 3);
    ansi_dim();
    std::cout << "Loss type: MSE  |  squeeze_max: " << squeezemax << "      ";
    ansi_reset();

    std::cout.flush();
  }

  ansi_show_cursor();

  auto end = nanos(); 
  auto opttime = (end - start) * 1e-9;
  std::cout <<"\n\n";
  std::cout << "Total training time: " << opttime  << '\n';
  std::cout << "Total EPOCHS: "        << epochmax << '\n'; 
  std::cout << "Learning rate: "       << std::fixed << std::setprecision(8) << eta << '\n'; 
  std::cout << "Training data size: "  << inputVals.size() << " data points" << '\n';
  std::cout << "Network size: "        << '\n'; 
  std::cout << "Linear  layer 1: " << inputVals.size() << " x " << 10000 << " neurons" << '\n';
  std::cout << "ReLu    layer 1: " << 10000            <<  "                  neurons" << '\n';
  std::cout << "Linear  layer 2: " << 10000            << " x " << 1000  << " neurons" << '\n'; 
  std::cout << "Sigmoid layer  : " << 1000             << "                   neurons" << '\n'; 
  std::cout << "Linear  layer 3: " << 1000             << " x " << 1     << " neurons" << '\n';
  std::cout << "Loss layer type: MSE LOSS\n"; 
  std::cout<<"\n";
}

void train_full_dataset_batched() {
  const std::size_t num_samples = 100; 
  const std::size_t input_dim   = 100; 
  const std::size_t hidden1_dim = 250;
  const std::size_t hidden2_dim = 250;
  const std::size_t output_dim  = 1;    
  const std::size_t batch_size  = 32;
  const std::size_t num_epochs  = 1500;
  const float       eta         = 0.00001f;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> x_dist(-3.0f, 3.0f);

  std::vector<float> dataset_X(num_samples * input_dim);
  std::vector<float> dataset_Y(num_samples * output_dim);

  for (std::size_t s = 0; s < num_samples; ++s) {
    for (std::size_t f = 0; f < input_dim; ++f) {
      dataset_X[s * input_dim + f] = x_dist(gen);
    }
    for (std::size_t o = 0; o < output_dim; ++o) {
      dataset_Y[s * output_dim + o] = 1.0f;
    }
  }

  neural::nn net;
  net.add_linear_relu_fused    (input_dim  , hidden1_dim);
  net.add_linear_relu_fused    (hidden1_dim, hidden2_dim);
  net.add_linear_sigmoid_fused (hidden2_dim, output_dim);
  net.add_loss                 (std::make_unique<neural::mse_loss>());

  init_console_utf8();

  TrainMetrics        metrics{};
  TrainDashboardConfig cfg{};
  cfg.num_samples   = (double)50000 * 50000;
  cfg.input_dim     = 50000;
  cfg.output_dim    = 10000;
  cfg.batch_size    = batch_size;
  cfg.num_epochs    = num_epochs;
  cfg.learning_rate = eta;
  cfg.loss_name     = "MSE";

  std::vector<LayerRow> layer_rows = {
      {"fc1", "Lin+ReLU",   "50000x50000",  (int)((50000 * 50000) / 10000ull), 0.0, 0.0},
      {"fc2", "Lin+ReLU",   "50000x10000",   (int)((50000  * 10000) / 10000ull), 0.0, 0.0},
      {"fc3", "Lin+Sigm",   "10000x1",     (int)((10000  * 1)   / 10000ull), 0.0, 0.0},
  };

  std::atomic<bool> running{true};

  std::thread dash_thread(
      dashboard_loop,
      std::cref(metrics),
      std::cref(cfg),
      std::cref(layer_rows),
      std::ref(running));

  std::vector<float> input_batch;    
  std::vector<float> target_batch;  
  std::vector<float> out_batch;      
  std::vector<float> grad_out_batch;   
  std::vector<float> grad_input_batch;  

  std::vector<std::size_t> indices(num_samples);
  std::iota(indices.begin(), indices.end(), 0);

  auto start_ns          = nanos();
  std::size_t total_seen = 0;
  double running_loss    = 0.0;
  double running_count   = 0.0;

  for (std::size_t epoch = 0; epoch < num_epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), gen);

    double epoch_loss_sum    = 0.0;
    std::size_t epoch_samples = 0;

    for (std::size_t batch_start = 0;
         batch_start < num_samples;
         batch_start += batch_size)
    {
      auto iter_start_ns = nanos();

      std::size_t curr_bs = std::min(batch_size, num_samples - batch_start);

      input_batch.resize(input_dim * curr_bs);
      target_batch.resize(output_dim * curr_bs);

      for (std::size_t b = 0; b < curr_bs; ++b) {
        std::size_t s = indices[batch_start + b];

        const float* x_src = &dataset_X[s * input_dim];
        const float* y_src = &dataset_Y[s * output_dim];

        for (std::size_t f = 0; f < input_dim; ++f) {
          input_batch[f * curr_bs + b] = x_src[f];
        }
        for (std::size_t o = 0; o < output_dim; ++o) {
          target_batch[o * curr_bs + b] = y_src[o];
        }
      }

      out_batch.clear();
      net.forward_batched(input_batch, curr_bs, out_batch);

#if DEBUG
      if (out_batch.size() != output_dim * curr_bs) {
        CPPDL_FATAL("out_batch size mismatch in train_full_dataset_batched");
      }
#endif

      float batch_loss = net.get_loss_batched(target_batch, curr_bs);
      epoch_loss_sum   += static_cast<double>(batch_loss) * static_cast<double>(curr_bs);
      epoch_samples    += curr_bs;

      running_loss  += static_cast<double>(batch_loss) * static_cast<double>(curr_bs);
      running_count += static_cast<double>(curr_bs);

      total_seen    += curr_bs;

      grad_out_batch = net.get_grad_batched(target_batch, curr_bs);

      net.backwards_batched(grad_out_batch, curr_bs, grad_input_batch);

      net.update(eta);

      auto iter_end_ns = nanos();
      double iter_s    = (iter_end_ns - iter_start_ns) * 1e-9;
      double elapsed_s = (iter_end_ns - start_ns)      * 1e-9;

      metrics.epoch          = static_cast<int>(epoch);
      metrics.batch_size     = static_cast<int>(curr_bs);
      metrics.loss           = batch_loss;
      metrics.avg_loss       = (running_count > 0.0) ? (running_loss / running_count) : batch_loss;
      metrics.train_time_s   = elapsed_s;

      double sps = (elapsed_s > 0.0)
                 ? static_cast<double>(total_seen) / elapsed_s
                 : 0.0;
      metrics.samples_per_sec = sps;

      float first_out    = out_batch.empty()    ? 0.0f : out_batch[0];
      float first_target = target_batch.empty() ? 0.0f : target_batch[0];

      metrics.last_out    = first_out;
      metrics.last_target = first_target;

      double ops_per_sample =
          2.0 * (double(input_dim)   * double(hidden1_dim) +
                 double(hidden1_dim) * double(hidden2_dim) +
                 double(hidden2_dim) * double(output_dim));
      double total_ops = ops_per_sample * double(curr_bs);
      double gflops    = (iter_s > 0.0) ? (total_ops / iter_s * 1e-9) : 0.0;

      metrics.gemm_m      = static_cast<int>(input_dim);
      metrics.gemm_n      = static_cast<int>(curr_bs);
      metrics.gemm_k      = static_cast<int>(hidden1_dim);
      metrics.gemm_gflops = gflops;

      //printf("\033[47;30m | EPOCH = %3zu\033[m", epoch);
      //printf("\033[47;30m | BATCH_LOSS = %f\033[m", batch_loss);
      //printf("\033[47;30m | OUTPUT[0] = %f\033[m", first_out);
      //printf("\033[47;30m | TARGET[0] = %f\033[m", first_target);
      //std::cout << " [ ";
      //net.draw_load_bar(static_cast<int>(epoch));
    }

    double epoch_loss = epoch_loss_sum / static_cast<double>(epoch_samples);
    (void)epoch_loss;
  }
  auto end_ns = nanos();
  double total_time = (end_ns - start_ns) * 1e-9;

  running = false;
  dash_thread.join();

  std::cout << "\n\n";
  std::cout << "Total training time: " << total_time   << " s\n";
  std::cout << "Total EPOCHS: "        << num_epochs   << '\n';
  std::cout << "Learning rate: "       << eta          << '\n';
  std::cout << "Batch size: "          << batch_size   << " (last batch may be smaller)\n";
  std::cout << "Num samples: "         << num_samples  << '\n';
  std::cout << "Input dim: "           << input_dim    << '\n';
  std::cout << "Output dim: "          << output_dim   << '\n';
  std::cout << "Loss type: MSE\n";
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

int main(){
  train_full_dataset_batched();
  //neural_network_test();
  //tokenizer_test(); 
}
