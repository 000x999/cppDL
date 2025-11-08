#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <cstdint> 
#include <vector>
#include "../logger_core/logger.h"
#include "../CRUSHBLAS_MODULE/core/include/matrix_core/matrix.hpp"
#include "../../defines.h"
  
/*
float conv_twod(const mat::mat_ops &x, const mat::mat_ops &kernel, uint8_t padding){
  const uint8_t ir = x.mat.m_row; 
  const uint8_t ic = x.mat.m_col;
  const uint8_t kr = kernel.mat.m_row;
  const uint8_t kc = kernel.mat.m_col; 
  if(ir < kr) throw std::invalid_argument("ERR");
  if(ic < kc) throw std::invalid_argument("ERR");
  const uint8_t rows = ir - kr + 2 * padding + 1; 
  const uint8_t cols = ic - kc + 2 * padding + 1; 
  mat::matrix res(rows, cols); 
  mat::mat_ops res_op(res); 
  res_op.zero_mat();
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
        auto in_t = mat::mat_ops::block_matrix(x,in_i, in_j, size_i, size_j);
        auto in_k = mat::mat_ops::block_matrix(kernel, k_i, k_j, size_i, size_j);
        res(i,j) = (in_t * in_k).sum();
      }
    }
    
  }
  return res; 
};
*/

#endif
