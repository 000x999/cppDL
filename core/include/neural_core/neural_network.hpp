#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <iostream> 
#include <memory>
#include <vector>
#include <utility>
#include <cmath>
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
  size_t layer_input_size; 
  size_t layer_output_size;
  std::vector<float> weights; 
  std::vector<float> biases;
  
  std::vector<float> input; 
  std::vector<float> output;
  
  std::vector<float> layer_deriv_in; 
  std::vector<float> layer_deriv_out; 

  std::vector<float> weight_grad;
  std::vector<float> bias_grad;
  layer_data(size_t input, size_t output);   
};

//Parent base for all derivations of layer
class layer{ 
public:
  virtual ~layer() = default; 
  virtual void forward(const std::vector<float> &layer_input_activations, layer_data &data) = 0;
  virtual void backwards(const std::vector<float> &layer_deriv_out, layer_data &data) = 0; 
  virtual void update (layer_data &data, float eta) {}; 
};

class linear : public layer{
public:
  void forward(const std::vector<float> &layer_input_activations, layer_data &data) override; 
  void backwards(const std::vector<float> &layer_deriv_out, layer_data &data) override;
  void update(layer_data &data, float eta) override;
};

class relu: public layer{
public:
  void forward(const std::vector<float> &layer_input_activations, layer_data &data)override;
  void backwards(const std::vector<float> &layer_deriv_out, layer_data &data)override; 
};

class sigmoid : public layer{
public:
  void forward(const std::vector<float> &layer_input_activations, layer_data &data)override;  
  void backwards(const std::vector<float> &layer_deriv_out, layer_data &data)override;
};

//Parent base for all derivations of loss
class loss {
  public:
  virtual ~loss() = default; 
  virtual float forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals) = 0; 
  virtual void backwards(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out) = 0; 
};

class mse_loss : public loss{
public:
  float forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals) override;
  void backwards(const std::vector<float> &layer_preds,const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out) override;
};

class cross_entropy_loss : public loss{
public:
  float forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals) override;
  void backwards(const std::vector<float> &layer_preds,const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out) override;
};

class nn {
public:
  std::vector<layer_data> data_layer;
  std::vector<std::unique_ptr<layer>> layers; 
  std::unique_ptr<loss> layer_loss_function; 

  void add_linear(size_t input, size_t output);
  void add_relu(size_t input);
  void add_sigmoid(size_t input);
  void add_loss(std::unique_ptr<loss> loss_in); 
  static void write_to_network(neural::nn &net_in); 
  neural::nn write_to_network(); 
  static void zero_grad(neural::nn &net_in);
  std::vector<float> forward(const std::vector<float> &layer_input_vals);
  void backwards(const std::vector<float> &layer_target_vals);
  float get_loss(const std::vector<float> &layer_target_vals);
  std::vector<float> get_grad(const std::vector<float> &layer_target_vals);
  void update(float eta);
  void draw_load_bar(int x);
  static uint64_t nanos(); 
};
};
#endif 
