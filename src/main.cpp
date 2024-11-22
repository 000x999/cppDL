#include "include/Tensor.h"

int main() {
  //First param is rank 
  //All the rest are dimension size 
  //If rank = 5 --> {x,x,x,x,x} and each dimension gets it's own size 
  Tensor::Tensor<float> tensor(5,{5,5,5,5,5}); 
    tensor.FillTensor();
    tensor.PrintTensor(); 
    
  return 0;
}
