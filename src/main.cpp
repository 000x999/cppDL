#include "include/Tensor.h"
#include "include/Functions.h"
#include "include/Structures.h"

int main() {
  //First param is rank 
  //All the rest are dimension size 
  //If rank = 5 --> {x,x,x,x,x} and each dimension gets it's own size 
  /*
  mat::matrix<float> D(10,10); 
  mat::matrix<float> A(9,9);
  int** arr; 
  A.zeros(); 
  A.displayMat();
  std::cout<<"\n";
  mat::matrix<float> C = A.block(3,3,3,3); 
  C.displayMat();
  std::cout<< "\n" << C.sum(); 
  */
  Tensor::Tensor<float> tensor(5,{5,5,5,5,5});  
  TensorOps::TensorOps<float> ops(tensor);
  ops.FillTensor();
  std::cout<<"\n=========== OG Tensor ===============\n"; 
  ops.PrintTensor();
  std::cout<<"\n=========== Tensor Mean ===============\n";
  std::cout<<Functions::Mean(ops.GetData());
  
  return 0;
}
