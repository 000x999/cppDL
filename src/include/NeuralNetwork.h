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
//Parent base for all derivations of layer
class layer{ 
  public:
  virtual ~layer() = default; 
  virtual void Forward(const std::vector<float> &inputVals) = 0;
  virtual void Backwards(const std::vector<float> &targetVals) = 0; 
  virtual void update (float eta) {}; 
};
  
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

class linear : public layer{
  public:
  void Forward(const std::vector<float> &inputVals) override{};
  void Backwards(const std::vector<float> &inputVals) override{};
  void update(float eta)override{}; 
};
  
class ReLU: public layer{
  public:
  void Forward(const std::vector<float> &inputVals)override{};
  void Backwards(const std::vector<float> &inputVals)override{};
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

