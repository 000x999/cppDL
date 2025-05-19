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
#include <cassert> 
#include "../tensor_core/tensor.h"
#include "../functions_core/functions.h"

namespace Neural{

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
  layer_data(size_t input, size_t output)
    :
    layer_input_size(input),
    layer_output_size(output),
    weights(input*output),
    biases(output),
    input(input,0.0f),
    output(output,0.0f),
    layer_deriv_in(input,0.0f),
    layer_deriv_out(output,0.0f),
    weight_grad(input*output, 0.0f),
    bias_grad(output,0.0f){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &w : weights) {
        w = dist(gen);
    }
    for (auto &b : biases) {
        b = dist(gen);
    } 
  }
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
  void forward(const std::vector<float> &layer_input_activations, layer_data &data) override{
    assert(layer_input_activations.size() == data.layer_input_size);
    data.input = layer_input_activations; 

    for(size_t i = 0; i < data.layer_output_size; ++i){
      float sum  = data.biases[i]; 
      for(size_t j = 0; j < data.layer_input_size; ++j){
       sum += data.weights[i * data.layer_input_size + j] * layer_input_activations[j];
      }
      data.output[i] = sum;
    }
  }

 void backwards(const std::vector<float> &layer_deriv_out, layer_data &data) override {
    assert(layer_deriv_out.size() == data.layer_output_size);
    std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f);
    std::fill(data.weight_grad.begin(), data.weight_grad.end(), 0.0f);
    std::fill(data.bias_grad.begin(), data.bias_grad.end(), 0.0f);

    for(size_t i = 0; i < data.layer_output_size; ++i){
        for(size_t j = 0; j < data.layer_input_size; ++j){
            data.layer_deriv_in[j] += data.weights[i * data.layer_input_size + j] * layer_deriv_out[i];
        }
    }
    for(size_t i = 0; i < data.layer_output_size; ++i){
        data.bias_grad[i] = layer_deriv_out[i];
        for(size_t j = 0; j < data.layer_input_size; ++j){
            data.weight_grad[i * data.layer_input_size + j] =
                layer_deriv_out[i] * data.input[j];
        }
    }
    data.layer_deriv_out = layer_deriv_out;
}

 void update(layer_data &data, float eta) override {
    for(size_t i = 0; i < data.layer_output_size; ++i){
        data.biases[i] -= eta * data.bias_grad[i];
        for(size_t j = 0; j < data.layer_input_size; ++j){
            data.weights[i * data.layer_input_size + j] -= 
                eta * data.weight_grad[i * data.layer_input_size + j];
        }
    }
}

};

class RELU: public layer{
  public:
  void forward(const std::vector<float> &layer_input_activations, layer_data &data)override{
    assert(layer_input_activations.size() == data.layer_input_size); 
    data.input = layer_input_activations;
    for(size_t i = 0; i < data.layer_output_size; ++i){
      data.output[i] = std::max(0.0f, layer_input_activations[i]);
    }
  }
  
  void backwards(const std::vector<float> &layer_deriv_out, layer_data &data)override{
    assert(layer_deriv_out.size() == data.layer_output_size); 
    std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f);
    for(size_t i = 0; i < data.layer_output_size; ++i){
      float grad = (data.input[i] > 0.0f) ? 1.0f : 0.0f; 
      data.layer_deriv_in[i] = layer_deriv_out[i] * grad; 
    }
    data.layer_deriv_out = layer_deriv_out; 
  }
};

class SIGMOID : public layer{
  public:
  void forward(const std::vector<float> &layer_input_activations, layer_data &data)override{
    assert(layer_input_activations.size() == data.layer_input_size);     
    data.output = Functions::Sigmoid(layer_input_activations); 
  }
  void backwards(const std::vector<float> &layer_deriv_out, layer_data &data)override{
    assert(layer_deriv_out.size() == data.layer_output_size); 
    std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f); 
    for(size_t i = 0; i < data.layer_output_size; ++i){
      float v = data.output[i]; 
      float dv = v * (1.0f - v); 
      data.layer_deriv_in[i] = layer_deriv_out[i] * dv; 
    }
    data.layer_deriv_out = layer_deriv_out; 
  }
};

class loss {
  public:
  virtual ~loss() = default; 
  virtual float forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals) = 0; 
  virtual void backwards(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out) = 0; 
};

class MSELOSS : public loss{
  public:
  float forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals) override {
    assert(layer_preds.size() == layer_target_vals.size());
    float loss = 0.0f; 
    for(size_t i = 0; i < layer_preds.size(); ++i){
      float diff = layer_preds[i] - layer_target_vals[i]; 
      loss += 0.5f * diff * diff; 
    }
    return loss;
  }
  void backwards(const std::vector<float> &layer_preds,const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out) override{
    assert(layer_preds.size() == layer_target_vals.size()); 
    layer_deriv_out.resize(layer_preds.size()); 
    for(size_t i = 0; i < layer_preds.size(); ++i){
      layer_deriv_out[i] = layer_preds[i] - layer_target_vals[i]; 
    } 
  }
};

class CrossEntropyLoss : public loss{
  public:
  float forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals) override{
    assert(layer_preds.size() == layer_target_vals.size());
    float loss = 0.0f;
    float epsi = 1e-15; 
    for(size_t i = 0; i < layer_preds.size(); ++i){
      float temp = std::max(layer_preds[i], epsi); 
      loss -= layer_target_vals[i] * std::log(temp); 
    }
    return loss; 
  }
  void backwards(const std::vector<float> &layer_preds,const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out) override{
    assert(layer_preds.size() == layer_target_vals.size()); 
    layer_deriv_out.resize(layer_preds.size()); 
    for(size_t i = 0; i < layer_preds.size(); ++i){
      layer_deriv_out[i] = layer_preds[i] - layer_target_vals[i]; 
    }
  }
};

class nn {
  public:
  std::vector<layer_data> layer_data;
  std::vector<std::unique_ptr<layer>> layers; 
  std::unique_ptr<loss> layer_loss_function; 

  void add_linear(size_t input, size_t output){
    layer_data.emplace_back(input, output); 
    layers.push_back(std::make_unique<linear>()); 
  }

  void add_RELU(size_t input){
    layer_data.emplace_back(input,input); 
    layers.push_back(std::make_unique<RELU>()); 
  }

  void add_SIGMOID(size_t input){
    layer_data.emplace_back(input, input); 
    layers.push_back(std::make_unique<SIGMOID>()); 
  }

  void add_loss(std::unique_ptr<loss> lossIn){
    layer_loss_function = std::move(lossIn); 
  }
  
  std::vector<float> forward(const std::vector<float> &layer_input_vals){
    assert(!layers.empty());
    std::vector<float> current = layer_input_vals; 
    for(size_t i = 0; i < layers.size(); ++i){
      layers[i]->forward(current, layer_data[i]); 
      current = layer_data[i].output; 
    }
    return current; 
  }
  
  void backwards(const std::vector<float> &layer_target_vals){
    std::vector<float> grad = layer_target_vals; 
    for(int i = static_cast<int>(layers.size() -1); i >= 0; --i){
      layers[i]->backwards(grad, layer_data[i]); 
      grad = layer_data[i].layer_deriv_in; 
    }
  }

  float get_loss(const std::vector<float> &layer_target_vals){
    if(!layer_loss_function){
      std::cerr << "NO LOSS ATTACHED\n"; 
      return 0.0f;
    }
    return layer_loss_function->forward(layer_data.back().output, layer_target_vals);
  }

  std::vector<float> get_grad(const std::vector<float> &layer_target_vals){
    if(!layer_loss_function){
      std::cerr <<"NO LOSS ATTACHED\n"; 
      return {}; 
    }
    std::vector<float> layer_deriv_out; 
    layer_loss_function->backwards(layer_data.back().output, layer_target_vals, layer_deriv_out);
    return layer_deriv_out;
  }

  void update(float eta){
    for(size_t i = 0; i < layers.size(); ++i){
      layers[i]->update(layer_data[i], eta); 
    }
  }
  
  void draw_load_bar(int x){
   float prog = (float)x / 2500; 
   float bw = 90;
   float pos = bw * prog;
   for(size_t i = 0; i < bw; ++i){
    if(i < pos) std::cout << "\033[35;106m \033[m"; 
    else std::cout << " "; 
  }
  std::cout<<"] " << prog * 100 << "%\r" << std::flush;
}

};
};
#endif 
