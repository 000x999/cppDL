#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <cmath>
#include <math.h>
#include <functional>
#include <numeric>
#include <algorithm>
#include <vector>

namespace Functions{
//Softmax
template <typename T>
inline std::vector<T> Softmax(std::vector<T> val_in, T beta){
  std::vector<T> exponentiated; 
  std::vector<T> res;
  for(size_t i = 0; i < val_in.size(); i++){
    exponentiated.emplace_back(exp(beta*val_in[i]));   
  }   
  for(size_t i = 0; i < exponentiated.size(); i++){
    res.emplace_back(exponentiated[i]/std::reduce(exponentiated.begin(), exponentiated.end()));
  }
  return res; 
}
//RELU
template <typename T>
inline T RELU(T val_in){
  if(val_in > 0){
  return (val_in + std::abs(val_in)) * 0.5f;
  }else{
    return 0; 
  }
}
//Sigmoid
template <typename T>
inline T Sigmoid(T val_in){
  return 1.0f / (1.0f + std::expf(-val_in)); 
}
//SiLU
template <typename T>
inline T SILU(T val_in){
  return val_in * Sigmoid(val_in);
}
//mish
template <typename T> 
inline T Mish(T val_in){
  return val_in*tanh(Softplus(val_in)); 
}
//softplus
template <typename T>
inline T Softplus(T val_in){
  return std::log(1+exp(val_in));
}
//squareplus 
template <typename T> 
inline T Squareplus(T val_in, T hyperparam){
  return (val_in + sqrtf((val_in*val_in) + hyperparam)) * 0.5f;
}
//binary step
template <typename T>
inline T BinStep(T val_in){
  if(val_in < 0){
    //deactivate node
    return 0; 
  }else{
    //activate node;
    return 1;
  }
}
//Mean
template <typename T>
inline T Mean(std::vector<T> val_in){
  return (std::reduce(val_in.begin(), val_in.end())) / val_in.size();
}
//SSE
template <typename T>
inline T SSE(std::vector<T> val_in){
  T res; 
  T mean = Mean(val_in); 
  for(size_t i =0; i < val_in.size(); i++){
   res = res + ((val_in[i] - mean)*(val_in[i] - mean)); 
  }
  return res; 
}
//feed forward
//backprop
//gradient descent 
}
#endif
