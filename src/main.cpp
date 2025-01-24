#include "include/Tensor.h"
#include "include/Functions.h"
#include "include/Structures.h"
#include "include/NeuralNetwork.h"
int main() {
  //First param is rank 
  //All the rest are dimension size 
  //If rank = 5 --> {x,x,x,x,x} and each dimension gets it's own size 
  /*
  mat::matrix<float> D(5,5); 
  mat::matrix<float> A(5,5);
  mat::MatOps::MatOps<float> matops(A); 
  mat::MatOps::MatOps<float> matopsD(D); 
  std::cout<<"MATOPS A ZEROES"<<std::endl;
  matops.fillMat();
  mat::MatOps::MatOps<float> newOp = matops * matopsD;
  std::cout<<"NEW OP DISPLAY"<<std::endl;
  newOp.displayMat();
  std::cout<<"MATOPS A DISPLAY"<<std::endl;
  matops.displayMat();
  std::cout<<"\n";
  std::cout<< "\n" << matops.sum(); 
  mat::matrix<float> E = mat::MatOps::MatOps<float>::matmul(A,D);  
  */
  nn::Network net; 
  net.connectLayer<nn::Net>(3,5);
  net.connectLayer<nn::ReLU>();
  net.connectLayer<nn::Net>(5,1); 

  Tensor::Tensor<float> tensor(5,{10,10,10,10,10});  
  TensorOps::TensorOps<float> ops(tensor);
  ops.FillTensor();
  std::cout<<"\n=========== OG Tensor ===============\n"; 
  ops.PrintTensor();
  std::cout<<"\n=========== Tensor Mean ===============\n";
  std::cout<<Functions::Mean(ops.GetData());
  
  return 0;
}
