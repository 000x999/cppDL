#include "include/Tensor.h"
#include "include/Functions.h"
#include "include/Structures.h"
#include "include/NeuralNetwork.h"
#include <stdlib.h>
#include <x86intrin.h>

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void loadbar(int x){
  float prog = (float)x / 2500; 
  float bw = 70;
  float pos = bw * prog;
  for(size_t i = 0; i < bw; ++i){
    if(i < pos) std::cout << "\033[35;106m \033[m"; 
    else std::cout << " "; 
  }
  std::cout<<"] " << prog * 100 << "%\r" << std::flush;
  
}

void MatMulBenchmark(float A, int blocksize){
  double totalOps = 2.0 * double(A) * double(A) * double(A);
  double gflopFactor = 1.0e-9;
  std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl; 
  mat::matrix<float> mat1(A, A);
  mat::matrix<float> mat2(A, A); 
  mat::MatOps<float> op1(mat1); 
  mat::MatOps<float> op2(mat2);
  op1.fillMat(); 
  op2.fillMat(); 
  op1.setBlockSize(blocksize); 
  auto start = nanos(); 
  mat::MatOps<float> op3 = op1 * op2; 
  auto end = nanos(); 
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX MatMul: " << optTime
              << "s, GFLOP/S = " << optGflops << "\n";
}

void TransposeBenchmark(float A){
  double totalOps =  double(A) * double(A);
  double memfactor = 2.0 * A *  A * sizeof(float);
  double memfactorgb = memfactor / (1024.0 * 1024.0 * 1024.0); 
  std::cout<< totalOps * 1e-6<< " KB" << std::endl; 
  mat::matrix<float> mat1(A, A);
  mat::MatOps<float> op1(mat1); 
  op1.fillMat();
  auto start = nanos();
  op1.TP();
  auto end = nanos(); 
  double optTime = (end - start) * 1e-9;
  double optmem =  memfactorgb / optTime;
  std::cout << "Transpose: " << optTime
              << "s, GB/S = " << optmem << "\n";
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
  size_t epochmax = 2500;
  size_t squeezemax = 10; 
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
      /*std::cout << "\033[47;30m EPOCH = \033[m" << epoch << "\033[47;30m | LOSS = \033[m" << loss << "\033[47;30m | OUTPUT[0] = \033[m" << out[0] 
       << "\033[47;30m | TARGET VAL = \033[m" << targetVals[0] << " [ ";*/
      printf("\033[47;30m | EPOCH = %i\033[m", (int)epoch);
      printf("\033[47;30m | LOSS = %f\033[m", loss);
      printf("\033[47;30m | OUTPUT[0] = %f\033[m", out[0]);
      printf("\033[47;30m | TARGET VAL = %i\033[m", (int)targetVals[0]);
      std::cout<<" [ ";
      loadbar(epoch); 
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

int main() {
  //NeuralTest();
 //MatMulBenchmark(1024, 16);
 TransposeBenchmark(8192*2);  
 //TensorBenchmark(); 
  return 0;
}
