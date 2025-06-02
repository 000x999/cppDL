#include "../../core/include/functions_core/functions.h"
#include "../../core/include/neural_core/neural_network.h"
#include "../../core/include/tokenizer_core/tokenizer.h"
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <x86intrin.h>

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void load_icon(){
  std::cout<<"\033[35;106m\r\033[m";
  fflush(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::cout<<"\033[35;106m\r\033[m";
  fflush(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::cout<<"\033[35;106m\r\033[m";
  fflush(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::cout<<"\033[35;106m\r\033[m";
  fflush(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void neural_network_test(){ 
  size_t epochmax = 100;
  size_t squeezemax = 1; 
  auto start = nanos(); 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-50.0f, 100.0f);
  std::vector<float> inputVals;
  for(size_t i = 0; i < 10000; ++i) {
    inputVals.emplace_back(dist(gen));
  }
  std::vector<float> targetVals = {1.0f};
  float eta = 0.000001; 
 
  neural::nn net;
  net.add_linear(inputVals.size(),1000); 
  net.add_relu(1000);
  net.add_linear(1000,100);
  net.add_sigmoid(100);
  net.add_linear(100,1);
  net.add_loss(std::make_unique<neural::mse_loss>());
  
  std::vector<float> out; 
  float loss; 
  
  for(size_t epoch = 0; epoch < epochmax; ++epoch){
    for(size_t squeeze = 0; squeeze < squeezemax; ++squeeze){
      out = net.forward(inputVals); 
      loss = net.get_loss(targetVals);
      auto derivOut  = net.get_grad(targetVals);
      net.backwards(derivOut); 
      net.update(eta);
    }
    if(epoch % 10 == 0){
      printf("\033[47;30m | EPOCH = %i\033[m", (int)epoch);
      printf("\033[47;30m | LOSS = %f\033[m", loss);  
      printf("\033[47;30m | OUTPUT[0] = %f\033[m", out[0]);
      printf("\033[47;30m | TARGET VAL = %i\033[m", (int)targetVals[0]);
      std::cout<<" [ ";
      net.draw_load_bar(epoch);
    }
  }
  auto end = nanos(); 
  auto opttime = (end - start) * 1e-9;
  std::cout<<"\n\n";
  std::cout<< "||| Total training time: " << opttime << std::endl; 
  std::cout<< "||| Total EPOCHS: " << epochmax <<std::endl; 
  std::cout<< "||| Total SQUEEZE: " << squeezemax << std::endl;
  std::cout<< "||| Training data size: " << inputVals.size() << " data points" <<std::endl; 
  std::cout<<"\n";
}

void tokenizer_test(){
  bpe::bpe_tokenizer tokenizer;
  std::string filePath = "src/TokenModels/DataSet.txt"; 
  std::ifstream infile {filePath};
  std::string trainingText {std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>()};
  size_t numMerges = 1000;
  std::cout << "Training BPE tokenizer with " << numMerges << " merges...\n";
  tokenizer.train(trainingText, numMerges);
  std::string testText = "I am testing out a large training data set for the tokenizer, we will see if this works properly.";
  std::vector<bpe::g_tokenid> encodedIds = tokenizer.encode(testText);
  std::cout << "Encoded IDs for test text:\n";
  int idCount = 0; 
  for (const auto& id : encodedIds) {
    std::cout << "Encoded ID: " << id << " -> '" << tokenizer.decode({id}) << "'\n";
    idCount++;
    if(idCount == 10){
      std::cout<<"The rest of the encoded ID's output here ...\n" << std::endl;
      break;
    }
  }
  std::string decodedText = tokenizer.decode(encodedIds);
  std::cout << "Decoded text: " << decodedText << std::endl;
  if (decodedText == testText) {
    std::cout << "***NOTE***: Encoding/decoding is lossless" << std::endl;
  } 
  else {
    std::cout << "***WARNING***: Encoding/decoding is not lossless" << std::endl;
  }
  tokenizer.save_model("src/TokenModels/bpe_vocab.txt", "src/TokenModels/bpe_merges.txt");
  std::cout << "Model saved to files" << std::endl;
  tokenizer.print_model_stats();
}

int main(){
  return 0;
}
