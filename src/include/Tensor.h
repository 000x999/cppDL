#ifndef TENSOR_H
#define TENSOR_H
#include "RandomGen.h"
#include <iostream> 
#include <vector>
#include <numeric>
#include <math.h>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <random>
#include <iomanip>
#include <cfloat>
#include <immintrin.h>
#include <cassert>
  
#define TENSOR
//ensure 4 byte allignment 
namespace Tensor{ 
#pragma pack(push,4)
template <typename T>
struct Tensor{
  size_t m_rank;
  std::vector<size_t> m_dimensions; 
  std::vector<T> m_data;
    Tensor(size_t rank, const std::vector<size_t>& dimensions)
        : m_rank(rank), m_dimensions(dimensions) {
        if (dimensions.size() != rank) {
            throw std::invalid_argument("Rank does not match dimensions.");
        }
        size_t totalSize = 1;
        for (const auto& dim : dimensions) {
            totalSize *= dim;
        }
        m_data = std::vector<T>(totalSize, T()); 
    }
  };
};

#pragma pack(pop)
namespace TensorOps{
  template <typename T>  
  class TensorOps{
    private:
    Tensor::Tensor<T> tensor; 
    public:
        TensorOps(Tensor::Tensor<T> &tensor)
          :tensor(tensor){}
    std::vector<T> GetData()const{return tensor.m_data;}; 
    size_t GetFlatIndex(const std::vector<size_t>& indices) const {
            assert(indices.size() == tensor.m_rank && "Indices must match tensor rank.");
            size_t flatIndex = 0;
            size_t stride = 1;

            for (int i = tensor.m_rank - 1; i >= 0; --i) {
                assert(indices[i] < tensor.m_dimensions[i] && "Index out of bounds.");
                flatIndex += indices[i] * stride;
                stride *= tensor.m_dimensions[i];
            }
            return flatIndex;
        }
        void ReshapeTensor(const std::vector<size_t>& reshape_in) {
          size_t newSize = 1;
          for (const auto& dim : reshape_in) newSize *= dim;
          assert(newSize == tensor.m_data.size() && "Reshape must not change total number of elements");
          tensor.m_dimensions = reshape_in;
        }

        T& ReturnAt(const std::vector<size_t>& indices) {
          return tensor.m_data[GetFlatIndex(indices)];
        }
      
        void SetElem(const std::vector<size_t>& indices, T val_in) {
          tensor.m_data[GetFlatIndex(indices)] = val_in;
        }

       void FillTensor(){
        static std::random_device rd; 
        static std::mt19937 gen(rd()); 
        static std::uniform_real_distribution<float> randVal(1, 10); 
        std::generate(tensor.m_data.begin(), tensor.m_data.end(), [&]() { return randVal(gen); });
        }

       void PrintTensor(size_t truncate = 5) const {
            size_t totalDims = tensor.m_dimensions.size();
            if (totalDims < 2) {
                std::cout << "Tensor rank too low to display properly." << std::endl;
                return;
            }
            auto printIndent = [](size_t level) {
                for (size_t i = 0; i < level; ++i) std::cout << "  ";
            };
            std::function<void(size_t, size_t, size_t)> printRecursive =
                [&](size_t currentDim, size_t offset, size_t level) {
                    if (currentDim == totalDims - 1) {
                        printIndent(level);
                        std::cout << "[";
                        for (size_t i = 0; i < tensor.m_dimensions[currentDim]; ++i) {
                            if (i >= truncate && i < tensor.m_dimensions[currentDim] - truncate) {
                                if (i == truncate) std::cout << "... ";
                                continue;
                            }
                            std::cout << std::fixed << std::setprecision(4) << tensor.m_data[offset + i];
                            if (i < tensor.m_dimensions[currentDim] - 1) std::cout << ", ";
                        }
                        std::cout << "]";
                    } else {
                        printIndent(level);
                        std::cout << "[\n";
                        for (size_t i = 0; i < tensor.m_dimensions[currentDim]; ++i) {
                            if (i >= truncate && i < tensor.m_dimensions[currentDim] - truncate) {
                                if (i == truncate) {
                                    printIndent(level + 1);
                                    std::cout << "...,\n";
                                }
                                continue;
                            }
                            printRecursive(currentDim + 1, offset + i * tensor.m_dimensions[currentDim + 1], level + 1);
                            if (i < tensor.m_dimensions[currentDim] - 1) std::cout << ",\n";
                        }
                        std::cout << "\n";
                        printIndent(level);
                        std::cout << "]";
                    }
                };
            std::cout << "tensor([\n";
            printRecursive(0, 0, 1);
            std::cout << "\n])" << std::endl;
        }
      //Tensor operations 
      std::vector<T> operator +(const Tensor::Tensor<T>& tensor_in) {
        std::vector<T> result; 
          for(size_t i = 0; i < this->tensor.m_data.size(); i++){
            result.emplace_back((this->tensor.m_data[i])+tensor_in.m_data[i]); 
          }
          return result; 
        } 
     };

      template <typename T>
      T dot(const Tensor::Tensor<T>& tensor_in, const Tensor::Tensor<T>& other_tensor){
        std::vector<T> tempVec; 
        T res = 0;   
        for(size_t i = 0; i < tensor_in.m_data.size(); i++){
          assert(tensor_in.m_data.size() == other_tensor.m_data.size() && "Tensor data set sizes must match");  
          tempVec.emplace_back(tensor_in.m_data[i]*other_tensor.m_data[i]);  
         }  
        return res = std::reduce(tempVec.begin(), tempVec.end());
      }
    }
#endif
