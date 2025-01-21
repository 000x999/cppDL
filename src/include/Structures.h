#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <immintrin.h>

namespace Vec{
#define VEC2
  template <typename T>
  struct alignas(sizeof(T)*2) Vec2{
  T m_x; 
  T m_y;  
  Vec2(T x_in, T y_in)
  : m_x(x_in), 
    m_y(y_in){}
  };
  namespace Vec2Ops{  
  template <typename T>    
  class Vec2Ops{
    private: 
    Vec::Vec2<T> vec2;
    public:
    Vec2Ops(Vec::Vec2<T> &vec2)
      :vec2(vec2){}
    Vec::Vec2<T> operator+(const Vec::Vec2<T>& rhs) const{return Vec2(vec2.m_x + rhs.m_x, vec2.m_y + rhs.m_y);}
    Vec::Vec2<T>& operator+=(const Vec::Vec2<T>& rhs){return *this = *this + rhs;}
    Vec::Vec2<T> operator*(float rhs) const{return Vec2(vec2.m_x * rhs, vec2.m_y * rhs);}
    Vec::Vec2<T>& operator*=(float rhs){return *this = *this * rhs;}
    Vec::Vec2<T> operator-(const Vec::Vec2<T>& rhs) const{return Vec2(vec2.m_x - rhs.m_x, vec2.m_y - rhs.m_y);}
    Vec::Vec2<T>& operator-=(const Vec::Vec2<T>& rhs){return *this = *this - rhs;}
    float fast_rsqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
    float fast_sqrt(float x) { return x * fast_rsqrt(x); }
    float GetLength(){return fast_sqrt(vec2.m_x*vec2.m_x + vec2.m_y*vec2.m_y);}
    float GetLengthSq() const{return vec2.m_x*vec2.m_x + vec2.m_y*vec2.m_y;}
    Vec::Vec2<T>& Normalize(){return *this = GetNormalized();}
    Vec::Vec2<T> GetNormalized(){const float len = GetLength(); if (len != 0.0f){return *this * (1.0f / len);} return *this;} 
    T dot(const Vec::Vec2<T>& vec2_in){return *this.m_x * vec2_in.m_x + *this.m_y * vec2_in.m_y;}
    };
  };
  
#define VEC3  
  template <typename T>
  struct alignas(sizeof(T)*3)Vec3{
    T m_x; 
    T m_y;
    T m_z;
    Vec3(T x_in, T y_in, T z_in)
     :
    m_x(x_in),
    m_y(y_in), 
    m_z(z_in){}
  };
  namespace Vec3Ops{
  template <typename T>
  class Vec3Ops{
    private:
    Vec::Vec3<T> vec3; 
    public:
      Vec3Ops(Vec::Vec3<T> &vec3)
        :vec3(vec3){}
      Vec::Vec3<T> operator+(const Vec::Vec3<T>& rhs) const{return Vec3(vec3.m_x + rhs.m_x, vec3.m_y + rhs.m_y, vec3.m_z+rhs.m_z);}
      Vec::Vec3<T>& operator+=(const Vec::Vec3<T>& rhs){return *this = *this + rhs;}
      Vec::Vec3<T> operator*(float rhs) const{return Vec3(vec3.m_x * rhs, vec3.m_y * rhs, vec3.m_z * rhs);}
      Vec::Vec3<T>& operator*=(float rhs){return *this = *this * rhs;}
      Vec::Vec3<T> operator-(const Vec::Vec3<T>& rhs) const{return Vec3(vec3.m_x - rhs.m_x, vec3.m_y - rhs.m_y, vec3.m_z - rhs.m_z);}
      Vec::Vec3<T>& operator-=(const Vec::Vec3<T>& rhs){return *this = *this - rhs;}
      float fast_rsqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
      float fast_sqrt(float x) { return x * fast_rsqrt(x); }
      float GetLength(){return fast_sqrt(vec3.m_x*vec3.m_x + vec3.m_y*vec3.m_y + vec3.m_z*vec3.m_z);}
      float GetLengthSq() const{return vec3.m_x*vec3.m_x + vec3.m_y*vec3.m_y + vec3.m_z*vec3.m_z;}
      Vec::Vec3<T>& Normalize(){return *this = GetNormalized();}
      Vec::Vec3<T> GetNormalized(){const float len = GetLength(); if (len != 0.0f){return *this * (1.0f / len);} return *this;} 
      T dot(const Vec::Vec3<T>& vec3_in){return *this.m_x * vec3_in.m_x + *this.m_y * vec3_in.m_y + *this.m_z * vec3_in.m_z;}   
  };
};

  #define VEC4  
  template <typename T>
  struct alignas(sizeof(T)*4)Vec4{
    T m_x; 
    T m_y;
    T m_z;
    T m_w;
    Vec4(T x_in, T y_in, T z_in, T w_in)
     :
    m_x(x_in),
    m_y(y_in), 
    m_z(z_in),
    m_w(w_in){} 
  };
  namespace Vec4Ops{
  template <typename T>
  class Vec4Ops{
    private:
    Vec::Vec4<T> vec4; 
    public:
      Vec4Ops(Vec::Vec4<T> &vec4)
        :vec4(vec4){}
      Vec::Vec4<T> operator+(const Vec::Vec4<T>& rhs) const{return Vec4(vec4.m_x + rhs.m_x, vec4.m_y + rhs.m_y, vec4.m_z+rhs.m_z);}
      Vec::Vec4<T>& operator+=(const Vec::Vec4<T>& rhs){return *this = *this + rhs;}
      Vec::Vec4<T> operator*(float rhs) const{return Vec4(vec4.m_x * rhs, vec4.m_y * rhs, vec4.m_z * rhs);}
      Vec::Vec4<T>& operator*=(float rhs){return *this = *this * rhs;}
      Vec::Vec4<T> operator-(const Vec::Vec4<T>& rhs) const{return Vec4(vec4.m_x - rhs.m_x, vec4.m_y - rhs.m_y, vec4.m_z - rhs.m_z, vec4.m_w - rhs.m_w );}
      Vec::Vec4<T>& operator-=(const Vec::Vec4<T>& rhs){return *this = *this - rhs;}
      float fast_rsqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
      float fast_sqrt(float x) { return x * fast_rsqrt(x); }
      float GetLength(){return fast_sqrt(vec4.m_x*vec4.m_x + vec4.m_y*vec4.m_y + vec4.m_z*vec4.m_z + vec4.m_w*vec4.m_w);}
      float GetLengthSq() const{return vec4.m_x*vec4.m_x + vec4.m_y*vec4.m_y + vec4.m_z*vec4.m_z + vec4.m_w*vec4.m_w;}
      Vec::Vec4<T>& Normalize(){return *this = GetNormalized();}
      Vec::Vec4<T> GetNormalized(){const float len = GetLength(); if (len != 0.0f){return *this * (1.0f / len);} return *this;} 
      T dot(const Vec::Vec4<T>& vec4_in){return *this.m_x * vec4_in.m_x + *this.m_y * vec4_in.m_y + *this.m_z * vec4_in.m_z + *this.m_w * vec4_in.m_w;}
    };
  };
}

#define MATRIX
namespace mat{
template <typename T>
struct matrix{
  size_t m_row; 
  size_t m_col;
  std::vector<std::vector<T>> mat; 
  matrix(size_t row_in, size_t col_in)
        : 
    m_row(row_in), 
    m_col(col_in),
    mat(row_in, std::vector<T>(col_in, 0)) {}
  };


  namespace MatOps{
  template <typename T>
  class MatOps{
    private:
    mat::matrix<T> mat; 
    public:
      MatOps(mat::matrix<T> &mat)
        :mat(mat){}
      void displayMat(){
       std::cout<<"["<<std::endl;
        for(size_t i = 0; i < mat.m_row; i++){
          for(size_t j = 0; j < mat.m_col; j++){
           std::cout << mat.mat[i][j]<<", ";   
          }
          std::cout<<""<<std::endl;
        }
       std::cout<<" ]";
      }

      void fillMat(){
        static std::random_device rd; 
        static std::mt19937 gen(rd()); 
        static std::uniform_real_distribution<float> randVal(0, 1);
        for(size_t i = 0; i < mat.m_row; i++){
          for(size_t j = 0; j < mat.m_col; j++){
          mat.mat[i][j] = randVal(gen);  
          }
        }
      }

      void zeros(){
        for(size_t i = 0; i < mat.m_row; i++){
          for(size_t j = 0; j < mat.m_col; j++){
          mat.mat[i][j] = 0;  
          }
        }
      }
      
  mat::matrix<T> matmul(const mat::matrix<T> &a, const mat::matrix<T> &b){
        matrix<T> tempMat(a.m_row, b.m_col); 
        if(a.m_col != b.m_row){
          std::cout<<"Invalid matrix sizes"<<std::endl;
        }
        for(size_t i = 0; i < tempMat.m_row; i++){
          for(size_t j = 0; j < tempMat.m_col; j++){
            for(size_t k = 0; k < mat.m_col; k++){
            tempMat.mat[i][j] += a.mat[i][k] * b.mat[k][j];
            }
          }
        }
        return tempMat;
      }
      
      T dot(const matrix<T>&b){
        T sum = 0; 
        if(mat.m_col != b.m_col){
          std::cout<<"Invalid matrix sizes"<<std::endl;
        }
        for(size_t i = 0; i < mat.m_row; i++ ){
          for(size_t j = 0; j < mat.m_col; j++){
            sum += mat.mat[i][j] * b.mat[i][j]; 
          }
        }
        return sum;
      }
      
    mat::matrix<T> block(size_t i,size_t j,size_t p,size_t q){
         if (i + p > this->mat.m_row || j + q > this->mat.m_col) {
            throw std::out_of_range("Block indices out of range");
        }
        matrix<T> tempMat(p, q);
        for(size_t a = 0; a < p; ++a){
          for(size_t b = 0; b < q; ++b){
            tempMat.mat[a][b] = mat.mat[a+i][b+j]; 
          }
        }
        return tempMat;
      }
     
      void TP(){
        matrix<T> tempMat(this->mat.m_row, this->mat.m_col); 
        for(size_t i = 0; i < this->mat.mat.m_row; i++){
          for(size_t j = 0; j < this->mat.mat.m_col; j++){
            tempMat.mat[i][j] = this->mat.mat[i][j]; 
          }
        }
        size_t temp = this->mat.m_row; 
        this->mat.m_row = this->mat.m_col; 
        this->mat.m_col = temp;
        for(size_t i = 0; i < this->mat.m_row; i++){
          for(size_t j = 0; j < this->mat.m_col; j++){
            this->mat.mat[i][j] = tempMat.mat[j][i]; 
          }
        }
      }

      T sum(){
        T res = 0;
        /*
        T res = std::accumulate(mat.begin(), mat.end(),0.0, 
              [](matrix<T> lhs, const matrix<T>& rhs){
                                return std::accumulate(rhs.mat.begin(), lhs.mat.end(), lhs);
                                });*/
        typename std::vector<std::vector<T>>::iterator row; 
        typename std::vector<T>::iterator col; 
        for(row = mat.begin(); row != mat.end(); ++row){
          for(col = row->begin(); col != row->end(); ++col){
              res += *col; 
          }
        }
        return res; 
      }

    mat::matrix<T> operator*(const mat::matrix<T> &rhs)const{
        matrix<T> tempMat(mat.m_row, rhs.m_col); 
        if(mat.m_col != rhs.m_row){
          std::cout<<"Invalid matrix sizes"<<std::endl;
        }
        for(size_t i = 0; i <tempMat.m_row; i++){
          for(size_t j = 0; j <tempMat.m_col; j++){
            for(size_t k = 0; k < mat.m_col; k++){
            tempMat.mat[i][j] += mat.mat[i][k] * rhs.mat[k][j];
            }
          }
        }
        return tempMat; 
      }

    mat::matrix<T>& operator*=(const mat::matrix<T> rhs)const{
        return *this = *this * rhs;
      }

    mat::matrix<T> operator+(const mat::matrix<T> &rhs)const{
        matrix<T> tempMat(mat.m_row, mat.m_col); 
        if(mat.m_row != rhs.m_row && mat.m_col != rhs.m_col){
          std::cout<<"Invalid matrix sizes"<<std::endl; 
        }
        for(size_t i = 0; i < tempMat.m_row; i++){
          for(size_t j = 0; j < tempMat.m_col; j++){
            tempMat.mat[i][j] = mat.mat[i][j]+rhs.mat[i][j];  
          }
        }
       return tempMat;  
      }

    mat::matrix<T>& operator+=(const mat::matrix<T> rhs)const{
        return *this = *this + rhs; 
      }

    mat::matrix<T> operator-(const mat::matrix<T> &rhs)const{
        matrix<T> tempMat(mat.m_row, mat.m_col); 
        if(mat.m_row != rhs.m_row && mat.m_col != rhs.m_col){
          std::cout<<"Invalid matrix sizes"<<std::endl; 
        }
        for(size_t i = 0; i < tempMat.m_row; i++){
          for(size_t j = 0; j < tempMat.m_col; j++){
            tempMat.mat[i][j] = mat.mat[i][j]-rhs.mat[i][j];  
          }
        }
       return tempMat;  
      }

    mat::matrix<T>& operator -=(const mat::matrix<T> rhs)const{
        return *this = *this - rhs; 
      }
  };
 };
}
#endif
