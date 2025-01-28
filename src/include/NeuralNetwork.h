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
#include "Tensor.h"
#include "Structures.h"
#include "Functions.h"

namespace Neural{

class layerData{
  public:
  size_t inputSize; 
  size_t outputSize;
  std::vector<float> weights; 
  std::vector<float> biases; 
  std::vector<float> activations;
  std::vector<float> grad;
  layerData(size_t input, size_t output)
    :
    inputSize(input),
    outputSize(output),
    weights(input*output, 0.0f),
    biases(output, 0.0f),
    activations(output,0.0f),
    grad(output,0.0f){}
};

//Parent base for all derivations of layer
class layer{ 
  public:
  virtual ~layer() = default; 
  virtual void Forward(const std::vector<float> &inputVals, layerData &data) = 0;
  virtual void Backwards(const std::vector<float> &targetVals, const std::vector<float> &gradOut, layerData &data) = 0; 
  virtual void update (layerData &data, float eta) {}; 
};

class linear : public layer{
  public:
  void Forward(const std::vector<float> &inputVals, layerData &data) override{
    for(size_t i = 0; i < data.outputSize; ++i){
      data.activations[i] = data.biases[i]; 
      for(size_t j = 0; j < data.inputSize; ++j){
        data.activations[i] += data.weights[i * data.inputSize + j] * inputVals[j];
      }
    }
  };

  void Backwards(const std::vector<float> &targetVals, const std::vector<float> &gradOut, layerData &data) override{
    for(size_t i = 0; i < data.outputSize; ++i){
      data.grad[i] = gradOut[i]; 
      for(size_t j = 0; j < data.inputSize; ++j){
        data.weights[i * data.inputSize + j] -= gradOut[i] * targetVals[j];
      }
      data.biases[i] -= gradOut[i];
    }
  };

  void update(layerData &data, float eta)override{
    for(auto &weights : data.weights){
      weights -= eta; 
    }
    for(auto &bias : data.biases){
      bias -= eta; 
    }
  };

};
  
class ReLU: public layer{
  public:
  void Forward(const std::vector<float> &inputVals, layerData &data)override{};
  void Backwards(const std::vector<float> &targetVals, const std::vector<float> &gradOut, layerData &data)override{};
};

class nn {
  public:
  std::vector<layerData> layerData;
  std::vector<std::unique_ptr<layer>> layers; 
  void addLinear(size_t input, size_t output){
    layerData.emplace_back(input, output); 
    layers.push_back(std::make_unique<linear>()); 
  }
  void addRelu(size_t input){
    layerData.emplace_back(input,input); 
    layers.push_back(std::make_unique<ReLU>()); 
  }
};

};




#endif 

