#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <cmath>
#include <math.h>
#include <functional>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>

namespace Functions{
//Softmax
template <typename T>
inline std::vector<T> Softmax(const std::vector<T>& val_in, T beta){
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
inline std::vector<T> RELU(const std::vector<T>& val_in){
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
inline std::vector<T> Sigmoid(const std::vector<T>& val_in){
  std::vector<T> tempVec; 
  for(auto& val: val_in){tempVec.emplace_back(1.0f / (1.0f + std::expf(-val)));} 
  return tempVec;
}
//SiLU
template <typename T>
inline std::vector<T> SILU(const std::vector<T>& val_in){
  std::vector<T> tempVec;
  std::vector<T> sigVec = Sigmoid(val_in);
  for(size_t i = 0; i < val_in.size(); i++){
   tempVec.emplace_back(val_in[i] * sigVec[i]);
  }
  return tempVec; 
}
//mish
template <typename T> 
inline std::vector<T> Mish(const std::vector<T>& val_in){
  std::vector<T> tempVec;
  std::vector<T> softVec = Softplus(val_in);
  for(size_t i = 0; i < val_in.size(); i++){
   tempVec.emplace_back(val_in[i]*tanh(softVec[i]));
  }
  return tempVec; 
}
//softplus
template <typename T>
inline std::vector<T> Softplus(const std::vector<T>& val_in){
  std::vector<T> tempVec; 
  for(size_t i = 0; i < val_in.size(); i++){
  tempVec.emplace_back(std::log(1+exp(val_in[i]))); 
  }
  return tempVec;
}
//squareplus 
template <typename T> 
inline std::vector<T> Squareplus(const std::vector<T>& val_in, T hyperparam){
  std::vector<T> tempVec;
  for(size_t i = 0; i <val_in.size(); i++){
    tempVec.emplace_back((val_in[i] + sqrtf((val_in[i]*val_in[i]) + hyperparam)) * 0.5f);
  }
  return tempVec;
}
//binary step
template <typename T>
inline std::vector<T> BinStep(const std::vector<T>& val_in){
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
inline T Mean(const std::vector<T>& val_in){
  T sum = std::accumulate(val_in.begin(), val_in.end(), 0.0); 
  T mean = sum / val_in.size(); 
  return mean;
}
//feed forward
//backprop
//gradient descent
//Calculate Out Gradients
//Calculate Hidden gradients 
}
#endif
