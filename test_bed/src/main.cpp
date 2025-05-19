#include "../../core/include/tensor_core/tensor.h"
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

void loadicon(){
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


void TensorBenchmark(){
  Tensor::Tensor<float> tensor1(3,{30,30,30});
  Tensor::Tensor<float> tensor2(3,{30,30,30});
  //Pass in previously created tensors through the TensorOp constructor to perform Tensor operations
  Tensor::TensorOps<float> ops1(tensor1);
  Tensor::TensorOps<float> ops2(tensor2);
  ops1.FillTensor();
  ops2.FillTensor();
  //Supports operator overloading and direct assigning to a new TensorOp
  Tensor::TensorOps<float> op3 = ops1*ops2; 
  //Zero's out all values in the Tensor
  //Prints Tensor formatted according to it's dimensionality
  op3.PrintTensor(); 
}

void NeuralTest(){ 
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
 
  Neural::nn net;
  net.addLinear(inputVals.size(),1000); 
  net.addRelu(1000);
  net.addLinear(1000,100);
  net.addSigmoid(100);
  net.addLinear(100,1);
  net.addLoss(std::make_unique<Neural::MSEloss>());
  
  std::vector<float> out; 
  float loss; 
  
  for(size_t epoch = 0; epoch < epochmax; ++epoch){
    for(size_t squeeze = 0; squeeze < squeezemax; ++squeeze){
      out = net.Forward(inputVals); 
      loss = net.getLoss(targetVals);
      auto derivOut  = net.getGrad(targetVals);
      net.Backwards(derivOut); 
      net.update(eta);
    }
    if(epoch % 10 == 0){
      printf("\033[47;30m | EPOCH = %i\033[m", (int)epoch);
      printf("\033[47;30m | LOSS = %f\033[m", loss);  
      printf("\033[47;30m | OUTPUT[0] = %f\033[m", out[0]);
      printf("\033[47;30m | TARGET VAL = %i\033[m", (int)targetVals[0]);
      std::cout<<" [ ";
      net.loadbar(epoch);
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

void tokenizerTest(){
  BPE::BPETokenizer tokenizer;
  std::string filePath = "src/TokenModels/DataSet.txt"; 
  std::ifstream infile {filePath};
  std::string trainingText {std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>()};
  size_t numMerges = 1000;
  std::cout << "Training BPE tokenizer with " << numMerges << " merges...\n";
  tokenizer.train(trainingText, numMerges);
  std::string testText = "I am testing out a large training data set for the tokenizer, we will see if this works properly.";
  std::vector<BPE::g_tokenid> encodedIds = tokenizer.encode(testText);
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
  tokenizer.saveModel("src/TokenModels/bpe_vocab.txt", "src/TokenModels/bpe_merges.txt");
  std::cout << "Model saved to files" << std::endl;
  tokenizer.printStats();
}

int main(){
  //NeuralTest();
  //TensorBenchmark();
  //tokenizerTest();
  return 0;
}
