#include "include/Tensor.h"
#include "include/Functions.h"
#include "include/Structures.h"

int main() {
  //First param is rank 
  //All the rest are dimension size 
  //If rank = 5 --> {x,x,x,x,x} and each dimension gets it's own size 
  mat::matrix<float> A(10,10); 
  mat::matrix<float> B(10,10);
  A.fillMat(); 
  B.fillMat(); 
  std::cout<<"\n=========== MAT A ===============\n";
  A.displayMat();
  std::cout<<"\n=========== MAT A ===============\n";
  std::cout<<"\n=========== MAT B ===============\n";
  B.displayMat(); 
  std::cout<<"\n=========== MAT B ===============\n";
  mat::matrix<float> C = A * B;
  std::cout<<"\n=========== MAT C ===============\n";
  C.displayMat(); 
  std::cout<<"\n=========== MAT C ===============\n";
  mat::matrix<float> D = C + A; 
  std::cout<<"\n=========== MAT D ===============\n";
  D.displayMat(); 
  std::cout<<"\n=========== MAT D ===============\n";

  /*
  Tensor::Tensor<float> tensor(5,{50,50,50,50,50});  
  tensor.FillTensor();
  std::cout<<"\n=========== OG Tensor ===============\n"; 
  tensor.PrintTensor();
  std::cout<<"\n=========== Tensor Mean ===============\n";
  std::cout<<Functions::Mean(tensor.m_data);
  */
  return 0;
}
