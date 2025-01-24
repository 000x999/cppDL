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

namespace nn{
  //Parent base for all derivations of layer
  class Layer {
  public:
    virtual ~Layer() = 0; 
    virtual std::vector<float> Forward(const std::vector<float> &inputVals) = 0;
    virtual std::vector<float> Backwards(const std::vector<float> &targetVals) = 0; 
    virtual void update (float eta) {}; 
  };

  class Net : Layer {
    private:
      std::vector<std::vector<float>> weights; 
      std::vector<float> biasNodes; 
      std::vector<float> outputNodes; 
  public:
    ~Net() override; 
     Net(size_t inputSize, size_t outputSize){
      static std::random_device rd; 
      static std::mt19937 gen(rd()); 
      static std::uniform_real_distribution<float> randVal(0, 1);
      weights.resize(outputSize, std::vector<float>(inputSize)); 
      biasNodes.resize(outputSize,0.0f);
      //Give the weights and biases random values
      for(auto& w: weights){
        for(auto& v: w){
          v = randVal(gen);
        }
      }
    }
    std::vector<float> Forward(const std::vector<float> &inputVals) override{}; 
    std::vector<float> Backwards(const std::vector<float> &inputVals) override{};
    void update(float eta)override{}; 
   };
  //Inherits the same methods as Layer except they have ReLU logic applied to them. 
  //This will be using the Functions.h methods as they are already implemented 
  //In place math can be done and will be tested alongside callable functions 
  class ReLU: Layer{
    public:
    std::vector<float> output; 
    std::vector<float> Forward(const std::vector<float> &inputVals)override{};
    std::vector<float> Backwards(const std::vector<float> &inputVals)override{};
    void update(float eta) override{};
  };

  class Network {
    public:
    std::vector<std::unique_ptr<Layer>> layers;
    Network() = default; 
    template <typename T, typename... Args>
    void connectLayer(Args&&... args){
      layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
    }
  };
  
};




#endif 

