#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <iostream> 
#include <memory>
#include <vector>
#include <utility>
#include <cmath>
#include <immintrin.h>
#include <string>           
#include <algorithm> 
#include <cstdlib> 
#include <random>
#include "cppdl_defines.h"
#include "tensor_core/tensor.hpp"
#include "logger_core/logger.hpp"
#include "level3/level3.hpp"
#include "neural_memory/nn_memory.hpp"

namespace neural{

struct alloc_pool{
  memory::neural_arena arena;
  explicit alloc_pool(size_t arena_size) : arena(arena_size) {}
}; 

struct neural_view{
  tens::tensor tensor; 
}; 

struct neural_weights{
  neural_view weights; 
  neural_view biases; 

  size_t input_features; 
  size_t output_features; 
}; 

//Parent base for all derivations of layer
class layer{ 
public:
  virtual                 ~layer            () = default; 
  virtual void            init              ( alloc_pool  &persistent_arena               ) = 0; 
  virtual neural_view     forward           ( neural_view &input_view, alloc_pool &arena  ) = 0; 
  virtual void            backwards         ( neural_view &grad_view , alloc_pool &arena  ) {} ;
  virtual neural_weights* get_weights       () { return nullptr; }
};

class linear : public layer{
private: 
  neural_weights weight_data; 
public:
                  linear      ( size_t input, size_t output                 ); 
  void            init        ( alloc_pool  &persistent_arena               ) override;
  neural_view     forward     ( neural_view &input_view, alloc_pool &arena  ) override; 
  neural_weights* get_weights ()                                              override { return &weight_data; }
};

class relu: public layer{
public:
  void        init    ( alloc_pool  &persistent_arena               ) override;
  neural_view forward ( neural_view &input_view, alloc_pool &arena  ) override; 
};

class sigmoid : public layer{
public:
  void        init    ( alloc_pool  &persistent_arena               ) override;
  neural_view forward ( neural_view &input_view, alloc_pool &arena  ) override; 
};
 
class nn {
public:
  std::vector<std::unique_ptr<layer>> layers; 
  
  void               add_linear               ( size_t input, size_t output                                     );
  void               add_relu                 (); 
  void               add_sigmoid              ();
  neural_view        forward                  ( const neural_view &layer_input_tensor, alloc_pool &arena        );
  float             *save_weights             (); 
  void               init                     ( alloc_pool        &persistent_arena                             ); 
  size_t             mem_reqs                 (); 
  static uint64_t    nanos                    ();
};
};
#endif 
