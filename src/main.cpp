#include "include/Tensor.h"

int main() {
  //First param is rank 
  //All the rest are dimension size 
  //If rank = 5 --> {x,x,x,x,x} and each dimension gets it's own size 
  Tensor::Tensor<float> tensor(5,{10,10,10,10,10}); 
  Tensor::Tensor<float> tensor1(5,{10,10,10,10,10});
  tensor.FillTensor();
  tensor1.FillTensor();
  std::cout<<"------------- Tensor 1 ---------------"<<std::endl;
  tensor.PrintTensor(); 
  std::cout<<"------------- Tensor 2 ---------------"<<std::endl;
  tensor1.PrintTensor();
  std::cout<<"------------- Tensor Addition ---------------"<<std::endl;
  Tensor::Tensor<float> tensor3(5,{10,10,10,10,10});  
  tensor3.m_data = tensor + tensor1;
  tensor3.PrintTensor();   
  std::cout<<"------------- Tensor Addition END ---------------\n\n"<<std::endl;
  std::cout<<"------------- Tensor Dot product ---------------"<<std::endl; 
  std::cout<<Tensor::dot(tensor, tensor1)<<std::endl;
  return 0;
}
