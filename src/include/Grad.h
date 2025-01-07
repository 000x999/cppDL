#ifndef GRAD_H
#define GRAD_H 
#include <vector>
#include "Structures.h"
#include "Functions.h"
#include "Tensor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <immintrin.h>
#include <algorithm>
#include <math.h>
#include <memory>
#include <cstdint> 
#include <iostream> 

#ifndef GRAD  
#define GRAD
  namespace grad{
  auto load_dataset = [](std::string dataFolder, const uint8_t padding){
   
  };
    
  template <typename T>    
  auto gradient = [](const mat::matrix<T> &xs,const mat::matrix<T> &ys,const mat::matrix<T> &ts,const uint8_t padding ){
    const uint8_t N = xs.size(); 
    const uint8_t R = xs.m_row;
    const uint8_t C = xs.m_col; 
    
    const int res_rows = xs.m_row - ys.m_row + 2 * padding + 1;
    const int res_cols = xs.m_col - ys.m_col + 2 * padding + 1;
    mat::matrix<T> res(res_rows, res_cols);
    res.zeros();
    for(size_t i = 0; i < N; ++i){
      const auto &X = xs.mat[i]; 
      const auto &Y = ys.mat[i];
      const auto &P = ts.mat[i];
      mat::matrix<T> delta = Y - P;
      mat::matrix<T> up = Conv2D(X, delta, padding); 
      res = res + up; 
    } 
    res *= 2.0/(R * C); 
    return res; 
  };    
  
  template <typename T> 
  auto gd = [](mat::matrix<T> &kernel, Dataset &dataset, const float lr, const uint8_t epochs){
    std::vector<T> losses; losses.reserve(epochs); 
    const uint8_t padding = kernel.m_row / 2; 
    const uint8_t N = dataset.size(); 
    
    std::vector<mat::matrix<T>> xs; xs.reserve(N); 
    std::vector<mat::matrix<T>> ys; ys.reserve(N);
    std::vector<mat::matrix<T>> ts; ts.reserve(N);
    uint8_t epoch = 0; 
    while(epoch<epochs){
      xs.clear(); 
      ys.clear();
      ts.clear();
      
      for(auto &instance : dataset){
        const auto & X = instance.first; 
        const auto & C = instance.second; 
        const auto Y = Conv2D(X, kernel, padding);
        xs.push_back(X);
        ts.push_back(C); 
        ys.push_back(Y);
      }
      losses.push_back(Functions::MSE(ys,ts));
      auto grad = gradient<T>(xs,ys,ts,padding);
      auto update = grad * lr; 
      kernel -= update; 
      epoch++; 
   }
  return losses; 
  };
} 
#endif
#endif
