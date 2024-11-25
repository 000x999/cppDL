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
inline std::vector<T> RELU(std::vector<T> val_in){
  std::vector<T> tempVec;
  for(size_t i = 0; i < val_in.size(); i++){
    if(val_in[i] > 0){
    tempVec.emplace_back((val_in[i] + std::abs(val_in[i])) * 0.5f);
    }else{
    tempVec.emplace_back(0); 
   }
  }
  return tempVec;
}
//Sigmoid
template <typename T>
inline std::vector<T> Sigmoid(std::vector<T> val_in){
  std::vector<T> tempVec; 
  for(auto& val: val_in){tempVec.emplace_back(1.0f / (1.0f + std::expf(-val)));} 
  return tempVec;
}
//SiLU
template <typename T>
inline std::vector<T> SILU(std::vector<T> val_in){
  std::vector<T> tempVec;
  std::vector<T> sigVec = Sigmoid(val_in);
  for(size_t i = 0; i < val_in.size(); i++){
   tempVec.emplace_back(val_in[i] * sigVec[i]);
  }
  return tempVec; 
}
//mish
template <typename T> 
inline std::vector<T> Mish(std::vector<T> val_in){
  std::vector<T> tempVec;
  std::vector<T> softVec = Softplus(val_in);
  for(size_t i = 0; i < val_in.size(); i++){
   tempVec.emplace_back(val_in[i]*tanh(softVec[i]));
  }
  return tempVec; 
}
//softplus
template <typename T>
inline std::vector<T> Softplus(std::vector<T> val_in){
  std::vector<T> tempVec; 
  for(size_t i = 0; i < val_in.size(); i++){
  tempVec.emplace_back(std::log(1+exp(val_in[i]))); 
  }
  return tempVec;
}
//squareplus 
template <typename T> 
inline std::vector<T> Squareplus(std::vector<T> val_in, T hyperparam){
  std::vector<T> tempVec;
  for(size_t i = 0; i <val_in.size(); i++){
    tempVec.emplace_back((val_in[i] + sqrtf((val_in[i]*val_in[i]) + hyperparam)) * 0.5f);
  }
  return tempVec;
}
//binary step
template <typename T>
inline std::vector<T> BinStep(std::vector<T> val_in){
  std::vector<T> tempVec; 
  for(size_t i =0; i < val_in.size(); i++){
    if(val_in[i] < 0){
      //deactivate node
      tempVec.emplace_back(0); 
   }else{
      //activate node;
     tempVec.emplace_back(1);
   }
  }
  return tempVec;
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
