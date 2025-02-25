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
  
  std::vector<float> input; 
  std::vector<float> output;
  
  std::vector<float> derivIn; 
  std::vector<float> derivOut; 

  std::vector<float> weightGrad;
  std::vector<float> biasGrad;
  layerData(size_t input, size_t output)
    :
    inputSize(input),
    outputSize(output),
    weights(input*output),
    biases(output),
    input(input,0.0f),
    output(output,0.0f),
    derivIn(input,0.0f),
    derivOut(output,0.0f),
    weightGrad(input*output, 0.0f),
    biasGrad(output,0.0f){
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
  virtual void Forward(const std::vector<float> &inputActivations, layerData &data) = 0;
  virtual void Backwards(const std::vector<float> &derivOut, layerData &data) = 0; 
  virtual void update (layerData &data, float eta) {}; 
};

class linear : public layer{
  public:
  void Forward(const std::vector<float> &inputActivations, layerData &data) override{
    assert(inputActivations.size() == data.inputSize);
    data.input = inputActivations; 

    for(size_t i = 0; i < data.outputSize; ++i){
      float sum  = data.biases[i]; 
      for(size_t j = 0; j < data.inputSize; ++j){
       sum += data.weights[i * data.inputSize + j] * inputActivations[j];
      }
      data.output[i] = sum;
    }
  }

 void Backwards(const std::vector<float> &derivOut, layerData &data) override {
    assert(derivOut.size() == data.outputSize);
    std::fill(data.derivIn.begin(), data.derivIn.end(), 0.0f);
    std::fill(data.weightGrad.begin(), data.weightGrad.end(), 0.0f);
    std::fill(data.biasGrad.begin(), data.biasGrad.end(), 0.0f);

    for(size_t i = 0; i < data.outputSize; ++i){
        for(size_t j = 0; j < data.inputSize; ++j){
            data.derivIn[j] += data.weights[i * data.inputSize + j] * derivOut[i];
        }
    }
    for(size_t i = 0; i < data.outputSize; ++i){
        data.biasGrad[i] = derivOut[i];
        for(size_t j = 0; j < data.inputSize; ++j){
            data.weightGrad[i * data.inputSize + j] =
                derivOut[i] * data.input[j];
        }
    }
    data.derivOut = derivOut;
}

 void update(layerData &data, float eta) override {
    for(size_t i = 0; i < data.outputSize; ++i){
        data.biases[i] -= eta * data.biasGrad[i];
        for(size_t j = 0; j < data.inputSize; ++j){
            data.weights[i * data.inputSize + j] -= 
                eta * data.weightGrad[i * data.inputSize + j];
        }
    }
}

};

class ReLU: public layer{
  public:
  void Forward(const std::vector<float> &inputActivations, layerData &data)override{
    assert(inputActivations.size() == data.inputSize); 
    data.input = inputActivations;
    for(size_t i = 0; i < data.outputSize; ++i){
      data.output[i] = std::max(0.0f, inputActivations[i]);
    }
  }
  
  void Backwards(const std::vector<float> &derivOut, layerData &data)override{
    assert(derivOut.size() == data.outputSize); 
    std::fill(data.derivIn.begin(), data.derivIn.end(), 0.0f);
    for(size_t i = 0; i < data.outputSize; ++i){
      float grad = (data.input[i] > 0.0f) ? 1.0f : 0.0f; 
      data.derivIn[i] = derivOut[i] * grad; 
    }
    data.derivOut = derivOut; 
  }
};

class Sigmoid : public layer{
  public:
  void Forward(const std::vector<float> &inputActivations, layerData &data)override{
    assert(inputActivations.size() == data.inputSize);     
    data.output = Functions::Sigmoid(inputActivations); 
  }
  void Backwards(const std::vector<float> &derivOut, layerData &data)override{
    assert(derivOut.size() == data.outputSize); 
    std::fill(data.derivIn.begin(), data.derivIn.end(), 0.0f); 
    for(size_t i = 0; i < data.outputSize; ++i){
      float v = data.output[i]; 
      float dv = v * (1.0f - v); 
      data.derivIn[i] = derivOut[i] * dv; 
    }
    data.derivOut = derivOut; 
  }
};

class loss {
  public:
  virtual ~loss() = default; 
  virtual float Forward(const std::vector<float> &preds, const std::vector<float> &targetVals) = 0; 
  virtual void Backwards(const std::vector<float> &preds, const std::vector<float> &targetVals, std::vector<float> &derivOut) = 0; 
};

class MSEloss : public loss{
  public:
  float Forward(const std::vector<float> &preds, const std::vector<float> &targetVals) override {
    assert(preds.size() == targetVals.size());
    float loss = 0.0f; 
    for(size_t i = 0; i < preds.size(); ++i){
      float diff = preds[i] - targetVals[i]; 
      loss += 0.5f * diff * diff; 
    }
    return loss;
  }
  void Backwards(const std::vector<float> &preds,const std::vector<float> &targetVals, std::vector<float> &derivOut) override{
    assert(preds.size() == targetVals.size()); 
    derivOut.resize(preds.size()); 
    for(size_t i = 0; i < preds.size(); ++i){
      derivOut[i] = preds[i] - targetVals[i]; 
    } 
  }
};

class CrossEntropyLoss : public loss{
  public:
  float Forward(const std::vector<float> &preds, const std::vector<float> &targetVals) override{
    assert(preds.size() == targetVals.size());
    float loss = 0.0f;
    float epsi = 1e-15; 
    for(size_t i = 0; i < preds.size(); ++i){
      float temp = std::max(preds[i], epsi); 
      loss -= targetVals[i] * std::log(temp); 
    }
    return loss; 
  }
  void Backwards(const std::vector<float> &preds,const std::vector<float> &targetVals, std::vector<float> &derivOut) override{
    assert(preds.size() == targetVals.size()); 
    derivOut.resize(preds.size()); 
    for(size_t i = 0; i < preds.size(); ++i){
      derivOut[i] = preds[i] - targetVals[i]; 
    }
  }
};

class nn {
  public:
  std::vector<layerData> layerData;
  std::vector<std::unique_ptr<layer>> layers; 
  std::unique_ptr<loss> lossFunc; 

  void addLinear(size_t input, size_t output){
    layerData.emplace_back(input, output); 
    layers.push_back(std::make_unique<linear>()); 
  }

  void addRelu(size_t input){
    layerData.emplace_back(input,input); 
    layers.push_back(std::make_unique<ReLU>()); 
  }

  void addSigmoid(size_t input){
    layerData.emplace_back(input, input); 
    layers.push_back(std::make_unique<Sigmoid>()); 
  }

  void addLoss(std::unique_ptr<loss> lossIn){
    lossFunc = std::move(lossIn); 
  }
  
  std::vector<float> Forward(const std::vector<float> &inputVals){
    assert(!layers.empty());
    std::vector<float> current = inputVals; 
    for(size_t i = 0; i < layers.size(); ++i){
      layers[i]->Forward(current, layerData[i]); 
      current = layerData[i].output; 
    }
    return current; 
  }
  
  void Backwards(const std::vector<float> &targetVals){
    std::vector<float> grad = targetVals; 
    for(int i = static_cast<int>(layers.size() -1); i >= 0; --i){
      layers[i]->Backwards(grad, layerData[i]); 
      grad = layerData[i].derivIn; 
    }
  }

  float getLoss(const std::vector<float> &targetVals){
    if(!lossFunc){
      std::cerr << "NO LOSS ATTACHED\n"; 
      return 0.0f;
     }
    return lossFunc->Forward(layerData.back().output, targetVals);
  }

  std::vector<float> getGrad(const std::vector<float> &targetVals){
    if(!lossFunc){
      std::cerr <<"NO LOSS ATTACHED\n"; 
      return {}; 
    }
    std::vector<float> derivOut; 
    lossFunc->Backwards(layerData.back().output, targetVals, derivOut);
    return derivOut;
  }

  void update(float eta){
    for(size_t i = 0; i < layers.size(); ++i){
      layers[i]->update(layerData[i], eta); 
    }
  }
};
};
#endif 
