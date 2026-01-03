#ifndef TENSOR_H 
#define TENSOR_H 
#include <vector>
#include <numeric>
#include <memory>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <random>
#include <cassert> 
#include <stdexcept> 
#include <limits>
#include <initializer_list>
#include "logger_core/logger.hpp"

namespace tens{
constexpr size_t max_dims = 4;

struct tensor_shape{
  size_t dims    [max_dims]; //size of each axis 
  size_t strides [max_dims]; //memory jump size, e.g. (tensor[x, k] --> index = x * strides[0] + k * strides[1])
/* 
  * access identifier
  * ndim tells us how many indices we need to access a single tensor elem 
*/ 
  int    ndim; 
  size_t numel        () const;
  bool   is_contiguous() const;
}; 

class tensor{
public: 
  float        *tensor_data; 
  tensor_shape  shape;

  tensor reshape   (std::initializer_list<size_t> reshape_size) const;
  tensor flatten   ()                                           const;
  tensor slice     (size_t slice_index)                         const;
  tensor transpose ()                                           const;
  tensor swizzle   (); 
  float* begin     ();
  float* end       (); 
  void   randn     (); 
};

class ops{
public:
  static __m512       fast_exp   (__m512 input_vec                                                         ); 
  static tens::tensor add        (const tens::tensor &left_tensor , const tens::tensor &right_tensor       );
  static tens::tensor add        (const tens::tensor &input_tensor, float scalar                           ); 
  static tens::tensor sub        (const tens::tensor &left_tensor , const tens::tensor &right_tensor       ); 
  static tens::tensor mul        (const tens::tensor &left_tensor , const tens::tensor &right_tensor       ); 
  static tens::tensor div        (const tens::tensor &left_tensor , const tens::tensor &right_tensor       );
  static tens::tensor scale      (const tens::tensor &input_tensor, float scale                            );
  static tens::tensor exp        (const tens::tensor &input_tensor                                         ); 
  static tens::tensor root       (const tens::tensor &input_tensor                                         );
  static tens::tensor tanh       (const tens::tensor &input_tensor                                         );
  static tens::tensor var        (const tens::tensor &input_tensor, size_t axis = -1, bool keep_dim = false);
  static tens::tensor sum        (const tens::tensor &input_tensor, size_t axis = -1, bool keep_dim = false); 
  static tens::tensor mean       (const tens::tensor &input_tensor, size_t axis = -1, bool keep_dim = false); 
  static tens::tensor max        (const tens::tensor &input_tensor, size_t axis = -1, bool keep_dim = false); 
  static tens::tensor min        (const tens::tensor &input_tensor, size_t axis = -1, bool keep_dim = false);
  static tens::tensor layer_norm (const tens::tensor &input_tensor, size_t axis = -1, float epsilon = 1e-5, float gamma = 0.0f, float beta = 1.0f); 
  static tens::tensor gelu       (const tens::tensor &input_tensor                                         );
  static tens::tensor softmax    (const tens::tensor &input_tensor, size_t axis = -1                       ); 
};
}; 

#endif 
