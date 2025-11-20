#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <iostream> 
#include <memory>
#include <vector>
#include <utility>
#include <cmath>
#include <immintrin.h>
#include "../../../core/include/CRUSHBLAS_MODULE/core/blas/level3/level3.hpp"
#include <string>           
#include <algorithm> 
#include <cstdlib> 
#include <random>
#include "../functions_core/functions.hpp"
#include "../logger_core/logger.hpp"
#include "../../defines.h"

namespace neural{

class layer_data{
public:
  size_t                layer_input_size; 
  size_t                layer_output_size;
  std::vector<float>    weights; 
  std::vector<float>    biases;
  
  std::vector<float>    input; 
  std::vector<float>    output;
  
  std::vector<float>    layer_deriv_in; 
  std::vector<float>    layer_deriv_out; 

  std::vector<float>    weight_grad;
  std::vector<float>    bias_grad;
                        layer_data
                        (size_t input, size_t output);
  std::vector<uint16_t> weights_fp16;
};

//Parent base for all derivations of layer
class layer{ 
public:

  virtual      ~layer            () = default; 
  virtual void forward           (const std::vector<float> &layer_input_activations, layer_data &data                                                          ) = 0;
  virtual void forward_batched   (const std::vector<float> &input_batch            , size_t batch_size, layer_data &data, std::vector<float> &output_batch     ) = 0; 
  virtual void backwards         (const std::vector<float> &layer_deriv_out        , layer_data &data                                                          ) = 0; 
  virtual void backwards_batched (const std::vector<float> &grad_output_batch      , size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch ) = 0; 
  virtual void update            (layer_data &data, float eta                                                                                                  ) {}; 
};

class linear : public layer{
public:

  void forward           (const std::vector<float> &layer_input_activations, layer_data &data                                                                  ) override; 
  void forward_batched   (const std::vector<float> &input_batch            , size_t batch_size, layer_data &data, std::vector<float> &output_batch             ) override;
  void backwards         (const std::vector<float> &layer_deriv_out        , layer_data &data                                                                  ) override;
  void backwards_batched (const std::vector<float> &grad_output_batch      , size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch         ) override; 
  void update            (layer_data &data                                 , float eta                                                                         ) override;
};

class linear_relu_fused : public layer{
public:

  void forward           (const std::vector<float> &layer_input_activations, layer_data &data                                                                  ) override;
  void forward_batched   (const std::vector<float> &input_batch            , size_t batch_size, layer_data &data, std::vector<float> &output_batch             ) override;
  void backwards         (const std::vector<float> &layer_deriv_out        , layer_data &data                                                                  ) override;
  void backwards_batched (const std::vector<float> &grad_output_batch      , size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch         ) override;
  void update            (layer_data &data                                 , float eta                                                                         ) override;
};
  
class linear_sigmoid_fused : public layer{
public:

  void forward           (const std::vector<float> &layer_input_activations, layer_data &data                                                                  ) override;
  void forward_batched   (const std::vector<float> &input_batch            , size_t batch_size, layer_data &data, std::vector<float> &output_batch             ) override;
  void backwards         (const std::vector<float> &layer_deriv_out        , layer_data &data                                                                  ) override;
  void backwards_batched (const std::vector<float> &grad_output_batch      , size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch         ) override;
  void update            (layer_data &data                                 , float eta                                                                         ) override;
};

class relu: public layer{
public:

  void forward           (const std::vector<float> &layer_input_activations, layer_data &data                                                                  ) override;
  void forward_batched   (const std::vector<float> &input_batch            , size_t batch_size, layer_data &data, std::vector<float> &output_batch             ) override; 
  void backwards         (const std::vector<float> &layer_deriv_out        , layer_data &data                                                                  ) override; 
  void backwards_batched (const std::vector<float> &grad_output_batch      , size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch         ) override;
};

class sigmoid : public layer{
public:

  void forward           (const std::vector<float> &layer_input_activations, layer_data &data                                                                  ) override; 
  void forward_batched   (const std::vector<float> &input_batch            , size_t batch_size, layer_data &data, std::vector<float> &output_batch             ) override;
  void backwards         (const std::vector<float> &layer_deriv_out        , layer_data &data                                                                  ) override;
  void backwards_batched (const std::vector<float> &grad_output_batch      , size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch         ) override;  
};

//Parent base for all derivations of loss
class loss {
  public:

  virtual       ~loss             () = default; 
  virtual float forward           (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals                                                                        ) = 0;
  virtual float forward_batched   (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, size_t num_outputs, size_t batch_size                                 ) = 0; 
  virtual void  backwards         (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out                                   ) = 0;
  virtual void  backwards_batched (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, size_t num_outputs, size_t batch_size, std::vector<float> &grad_preds ) = 0; 
};

class mse_loss : public loss{
public:

  float forward           (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals                                                                                ) override;
  float forward_batched   (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, size_t num_outputs, size_t batch_size                                         ) override;
  void  backwards         (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out                                           ) override;
  void  backwards_batched (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, size_t num_outputs, size_t batch_size, std::vector<float> &grad_preds         ) override;
};

class cross_entropy_loss_logits : public loss{
public:

  float forward           (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals                                                                                ) override;
  float forward_batched   (const std::vector<float> &logits_data, const std::vector<float> &layer_target_vals, size_t num_classes, size_t batch_size                                         ) override;
  void  backwards         (const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out                                           ) override;
  void  backwards_batched (const std::vector<float> &logits_data, const std::vector<float> &layer_target_vals, size_t num_classes, size_t batch_size, std::vector<float> &grad_logits        ) override;
};
 
class nn {
public:

  std::vector<layer_data>             data_layer;
  std::vector<std::unique_ptr<layer>> layers; 
  std::unique_ptr<loss>               layer_loss_function; 
  
  void               sync_weights_fp32_fp16   (layer_data &data); 
  void               add_linear               (size_t input, size_t output);
  void               add_linear_relu_fused    (size_t input_size, size_t output_size); 
  void               add_linear_sigmoid_fused (size_t input_size, size_t output_size); 
  void               add_relu                 (size_t input);
  void               add_sigmoid              (size_t input);
  void               add_loss                 (std::unique_ptr<loss> loss_in); 
  static void        zero_grad                (neural::nn &net_in);
  std::vector<float> forward                  (const std::vector<float> &layer_input_vals);
  std::vector<float> forward_batched          (const std::vector<float> &input_batch, size_t batch_size, std::vector<float> &output_batch);
  void               backwards                (const std::vector<float> &layer_target_vals);
  void               backwards_batched        (const std::vector<float> &grad_output_batch, size_t batch_size, std::vector<float> &grad_input_batch);
  float              get_loss                 (const std::vector<float> &layer_target_vals);
  float              get_loss_batched         (const std::vector<float> &target_batch, size_t batch_size); 
  std::vector<float> get_grad                 (const std::vector<float> &layer_target_vals);
  std::vector<float> get_grad_batched         (const std::vector<float> &target_batch, size_t batch_size); 
  void               update                   (float eta);
  void               draw_load_bar            (int x);
  static uint64_t    nanos                    ();

};
};
#endif 
