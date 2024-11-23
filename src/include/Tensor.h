#ifndef TENSOR_H
#define TENSOR_H
#include "glm/detail/qualifier.hpp"
#include "glm/ext/matrix_float2x2.hpp"
#include <cfloat>
#include <glm/glm.hpp>
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
namespace Tensor{ 
template <typename T>
struct Tensor{
  uint8_t m_rank;
  std::vector<size_t> m_dimensions; 
  std::vector<T> m_data;
  static const uint8_t fixed2x2 = 2;
  static const uint8_t fixed3x3 = 3;
  static const uint8_t fixed4x4 = 4;
  //Static pre-defined sizes for ease of use
  std::vector<T> rank1Tensor; 
  std::vector<glm::mat<fixed2x2, fixed2x2, T>> rank3Tensor2x2;
  std::vector<glm::mat<fixed3x3, fixed3x3, T>> rank3Tensor3x3;
  std::vector<glm::mat<fixed4x4, fixed4x4, T>> rank3Tensor4x4;
  //Dynamic Flat Tensor implementation
    Tensor(uint8_t rank, const std::vector<size_t>& dimensions)
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
    size_t GetFlatIndex(const std::vector<size_t>& indices) const {
        assert(indices.size() == m_rank && "Indices must match tensor rank.");
        size_t flatIndex = 0;
        size_t stride = 1;

        for (int i = m_rank - 1; i >= 0; --i) {
            assert(indices[i] < m_dimensions[i] && "Index out of bounds.");
            flatIndex += indices[i] * stride;
            stride *= m_dimensions[i];
        }
        return flatIndex;
    }
    void ReshapeTensor(const std::vector<size_t>& reshape_in) {
      size_t newSize = 1;
      for (const auto& dim : reshape_in) newSize *= dim;
      assert(newSize == m_data.size() && "Reshape must not change total number of elements");
      m_dimensions = reshape_in;
    }

    T& ReturnAt(const std::vector<size_t>& indices) {
      return m_data[GetFlatIndex(indices)];
    }
  
    void SetElem(const std::vector<size_t>& indices, T val_in) {
      m_data[GetFlatIndex(indices)] = val_in;
    }
   void FillTensor(){
    static std::random_device rd; 
    static std::mt19937 gen(rd()); 
    static std::uniform_real_distribution<float> randVal(-10000.100000, 300000.300000); 
    std::generate(m_data.begin(), m_data.end(), [&]() { return randVal(gen); });
  }
    void PrintTensor(size_t truncate = 5) const {
        size_t totalDims = m_dimensions.size();
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
                    for (size_t i = 0; i < m_dimensions[currentDim]; ++i) {
                        if (i >= truncate && i < m_dimensions[currentDim] - truncate) {
                            if (i == truncate) std::cout << "... ";
                            continue;
                        }
                        std::cout << std::fixed << std::setprecision(4) << m_data[offset + i];
                        if (i < m_dimensions[currentDim] - 1) std::cout << ", ";
                    }
                    std::cout << "]";
                } else {
                    printIndent(level);
                    std::cout << "[\n";
                    for (size_t i = 0; i < m_dimensions[currentDim]; ++i) {
                        if (i >= truncate && i < m_dimensions[currentDim] - truncate) {
                            if (i == truncate) {
                                printIndent(level + 1);
                                std::cout << "...,\n";
                            }
                            continue;
                        }
                        printRecursive(currentDim + 1, offset + i * m_dimensions[currentDim + 1], level + 1);
                        if (i < m_dimensions[currentDim] - 1) std::cout << ",\n";
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
  std::vector<T> operator +(const Tensor<T>& tensor_in) {
    std::vector<T> result; 
    for(auto& lhs: this->m_data){
      for(auto& rhs: tensor_in.m_data){
        result.emplace_back(lhs+rhs); 
        } 
      }
      return result; 
    }
   
  T dot(const Tensor<T>& tensor_in, const Tensor<T>& other_tensor)const{
    std::vector<T> tempVec; 
    T res = 0;   
    for(size_t i; i < tensor_in.m_data.size(); i++){
      assert(tensor_in.m_data.size() == other_tensor.m_data.size() && "Tensors are not the same size"); 
      if(tensor_in.m_data.size() != other_tensor.m_data.size){
        std::cout<<"Tensors are not the same size, dot product cannot be computed"<<std::endl;
        break; 
      }
      tempVec.emplace_back(tensor_in.m_data[i]*other_tensor.m_data[i]);  
     }  
    return res = std::reduce(tempVec.begin(), tempVec.end());
  }
 };
}
#endif
