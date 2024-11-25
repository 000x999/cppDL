#include "include/Tensor.h"
#include "include/Functions.h"
int main() {
  //First param is rank 
  //All the rest are dimension size 
  //If rank = 5 --> {x,x,x,x,x} and each dimension gets it's own size 
  Tensor::Tensor<float> tensor(3,{2,2,2}); 
  tensor.FillTensor();
  std::cout<<"\n=========== OG Tensor ===============\n"; 
  tensor.PrintTensor();
  tensor.m_data = Functions::Sigmoid(tensor.m_data); 
  std::cout<<"\n=========== Sigmoided Tensor ===============\n";
  tensor.PrintTensor();
  std::cout<<"\n=========== RELU tensor ===============\n";  
  tensor.m_data = Functions::RELU(tensor.m_data);
  tensor.PrintTensor();
  return 0;
}
