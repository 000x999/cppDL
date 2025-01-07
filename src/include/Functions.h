#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <cmath>
#include <math.h>
#include <functional>
#include <numeric>
#include "Structures.h"
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

template <typename T> 
inline T MSE(const std::vector<mat::matrix<T>>& in1, const std::vector<mat::matrix<T>>& in2){ 
  T err = 0; 
  size_t elems = 0; 
  for(size_t i = 0; i < in1.size(); ++i){
    const auto &mat1 = in1[i]; 
    const auto &mat2 = in2[i];
    for(size_t j = 0; j < mat1.size(); ++j){
      const auto &row1 = mat1[j]; 
      const auto &row2 = mat2[j]; 
      for(size_t k = 0; k < row1.size(); ++k){
        T diff = row1[k] - row2[k]; 
        err += diff * diff; 
        ++elems; 
      }
    }
  }
  return (err/elems); 
}
  
template <typename T>
inline T Conv2D(const mat::matrix<T> &x, const mat::matrix<T> &kernel, uint8_t padding){
  const uint8_t ir = x.m_row; 
  const uint8_t ic = x.m_col;
  const uint8_t kr = kernel.m_row;
  const uint8_t kc = kernel.m_col; 
  if(ir < kr) throw std::invalid_argument("ERR");
  if(ic < kc) throw std::invalid_argument("ERR");
  const uint8_t rows = ir - kr + 2 * padding + 1; 
  const uint8_t cols = ic - kc + 2 * padding + 1; 
  mat::matrix<T> res(rows, cols); 
  res.zeros();
  auto resizeDims = [&padding](uint8_t pos, uint8_t k, uint8_t len){
    uint8_t input = pos - padding;
    uint8_t kernel = 0; 
    uint8_t size = k; 
    if (input < 0){
      kernel = -input;
      size += input; 
      input = 0; 
    }
    if(input + size > len){
      size = len - input; 
    }
    return std::make_tuple(input, kernel, size);
  }; 
 
  for(size_t i = 0; i < rows; ++i){
    const auto[in_i, k_i, size_i] = resizeDims(i, kr,ir); 
    for(size_t j = 0; size_i > 0 && j < cols; ++j){
      const auto[in_j,k_j, size_j] = resizeDims(j,kc,ic);
      if(size_j > 0){
        auto in_t = x.block(in_i, in_j, size_i, size_j);
        auto in_k = kernel.block(k_i, size_i, size_j);
        res(i,j) = (in_t * in_k).sum();
      }
    }
    
  }
  return res; 
};
//feed forward
//backprop
//gradient descent
//Calculate Out Gradients
//Calculate Hidden gradients 
}
#endif
