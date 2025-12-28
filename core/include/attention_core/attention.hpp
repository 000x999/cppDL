#ifndef ATTENTION_H
#define ATTENTION_H
#include <iostream>
#include <algorithm>
#include <math.h>

#include "cppdl_defines.h"
#include "tensor_core/tensor.hpp"
#include "neural_memory/nn_memory.hpp"
#include "level3/level3.hpp"

namespace atten{

struct atten_weights{
  tens::tensor w_queries; 
  tens::tensor w_keys; 
  tens::tensor w_values; 
  tens::tensor w_output; 

  size_t input_features; 
  size_t output_features; 
};

struct atten_comps{
  tens::tensor queries; 
  tens::tensor keys;
  tens::tensor values; 
  tens::tensor attn_scores; 
  tens::tensor attn_output; 
}; 

struct atten_pool{
  memory::neural_arena arena;
  explicit atten_pool(size_t arena_size) : arena(arena_size){}
}; 

class attention{
private:
  atten_weights weights_data; 
  atten_comps   atten_data;
  size_t        embedded_dim; 
  size_t        num_heads; 
  size_t        head_dim;  

public:
  attention                 (size_t       embedded_dim  , size_t    num_heads     ); 
  void         init         (atten_pool   &persistent_arena                       ); 
  void         load_weights (float *w_q,  float *w_k, float *w_v, float *w_o      );
  tens::tensor forward      (tens::tensor &input_tensor , atten_pool &alloc_pool  ); 
};

class multi_head_attention{
private:
  atten_weights weights_data;
  atten_comps   atten_data; 
  size_t        embedded_dim; 
  size_t        num_heads; 
  size_t        head_dim; 

public: 
  multi_head_attention      (size_t       embedded_dim, size_t num_heads        ); 
  void         init         (atten_pool   &persistent_arena                     ); 
  void         load_weights (float *w_q,  float *w_k, float *w_v, float*w_o     ); 
  tens::tensor forward      (tens::tensor &input_tensor, atten_pool &alloc_pool ); 
};

};//namespace  
#endif 
