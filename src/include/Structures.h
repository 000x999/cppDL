#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <immintrin.h>

#pragma pack(push, 1)
namespace Vec{
  #define VEC2
  template <typename T>
  struct Vec2{
  T m_x; 
  T m_y;  
  Vec2(T x_in, T y_in)
  : m_x(x_in), 
    m_y(y_in){}
  Vec2 operator+(const Vec2& rhs) const{return Vec2(m_x + rhs.m_x, m_y + rhs.m_y);}
	Vec2& operator+=(const Vec2& rhs){return *this = *this + rhs;}
	Vec2 operator*(float rhs) const{return Vec2(m_x * rhs, m_y * rhs);}
	Vec2& operator*=(float rhs){return *this = *this * rhs;}
	Vec2 operator-(const Vec2& rhs) const{return Vec2(m_x - rhs.x, m_y - rhs.y);}
	Vec2& operator-=(const Vec2& rhs){return *this = *this - rhs;}
	float fast_rsqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
	float fast_sqrt(float x) { return x * fast_rsqrt(x); }
	float GetLength(){return fast_sqrt(m_x*m_x + m_y*m_y);}
	float GetLengthSq() const{return m_x*m_x + m_y*m_y;}
	Vec2& Normalize(){return *this = GetNormalized();}
	Vec2 GetNormalized(){const float len = GetLength(); if (len != 0.0f){return *this * (1.0f / len);} return *this;} 
  T dot(const Vec2& vec2_in){return *this.m_x * vec2_in.m_x + *this.m_y * vec2_in.m_y;}
};
#pragma pack(pop)

#pragma pack(push, 1)
  #define VEC3  
  template <typename T>
  struct Vec3{
    T m_x; 
    T m_y;
    T m_z;
    Vec3(T x_in, T y_in, T z_in)
     :
    m_x(x_in),
    m_y(y_in), 
    m_z(z_in){}
  Vec3 operator+(const Vec3& rhs) const{return Vec3(m_x + rhs.m_x, m_y + rhs.m_y, m_z+rhs.m_z);}
	Vec3& operator+=(const Vec3& rhs){return *this = *this + rhs;}
	Vec3 operator*(float rhs) const{return Vec3(m_x * rhs, m_y * rhs, m_z * rhs);}
	Vec3& operator*=(float rhs){return *this = *this * rhs;}
	Vec3 operator-(const Vec3& rhs) const{return Vec3(m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z);}
	Vec3& operator-=(const Vec3& rhs){return *this = *this - rhs;}
	float fast_rsqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
	float fast_sqrt(float x) { return x * fast_rsqrt(x); }
	float GetLength(){return fast_sqrt(m_x*m_x + m_y*m_y + m_z*m_z);}
	float GetLengthSq() const{return m_x*m_x + m_y*m_y + m_z*m_z;}
	Vec3& Normalize(){return *this = GetNormalized();}
	Vec3 GetNormalized(){const float len = GetLength(); if (len != 0.0f){return *this * (1.0f / len);} return *this;} 
  T dot(const Vec3& vec3_in){return *this.m_x * vec3_in.m_x + *this.m_y * vec3_in.m_y + *this.m_z * vec3_in.m_z;}  
};
#pragma pack(pop)

#pragma pack(push, 1)
  #define VEC4  
  template <typename T>
  struct Vec4{
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
  Vec4 operator+(const Vec4& rhs) const{return Vec4(m_x + rhs.m_x, m_y + rhs.m_y, m_z+rhs.m_z);}
	Vec4& operator+=(const Vec4& rhs){return *this = *this + rhs;}
	Vec4 operator*(float rhs) const{return Vec4(m_x * rhs, m_y * rhs, m_z * rhs);}
	Vec4& operator*=(float rhs){return *this = *this * rhs;}
	Vec4 operator-(const Vec4& rhs) const{return Vec4(m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z, m_w - rhs.m_w );}
	Vec4& operator-=(const Vec4& rhs){return *this = *this - rhs;}
	float fast_rsqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }
	float fast_sqrt(float x) { return x * fast_rsqrt(x); }
	float GetLength(){return fast_sqrt(m_x*m_x + m_y*m_y + m_z*m_z + m_w*m_w);}
	float GetLengthSq() const{return m_x*m_x + m_y*m_y + m_z*m_z + m_w*m_w;}
	Vec4& Normalize(){return *this = GetNormalized();}
	Vec4 GetNormalized(){const float len = GetLength(); if (len != 0.0f){return *this * (1.0f / len);} return *this;} 
  T dot(const Vec4& vec4_in){return *this.m_x * vec4_in.m_x + *this.m_y * vec4_in.m_y + *this.m_z * vec4_in.m_z + *this.m_w * vec4_in.m_w;}
  }; 
}
#pragma pack(pop)

#pragma pack(push, 4)
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
  
  void displayMat(){
   std::cout<<"["<<std::endl;
    for(size_t i = 0; i < m_row; i++){
      for(size_t j = 0; j < m_col; j++){
       std::cout << mat[i][j]<<", ";   
      }
      std::cout<<""<<std::endl;
    }
   std::cout<<" ]";
  }

  void fillMat(){
    static std::random_device rd; 
    static std::mt19937 gen(rd()); 
    static std::uniform_real_distribution<float> randVal(0, 1);
    for(size_t i = 0; i < m_row; i++){
      for(size_t j = 0; j < m_col; j++){
      mat[i][j] = randVal(gen);  
      }
    }
  }

  void zeros(){
    for(size_t i = 0; i < m_row; i++){
      for(size_t j = 0; j < m_col; j++){
      mat[i][j] = 1;  
      }
    }
  }
  
  matrix matmul(const matrix<T> &a, const matrix<T> &b){
    matrix<T> tempMat(a.m_row, b.m_col); 
    if(a.m_col != b.m_row){
      std::cout<<"Invalid matrix sizes"<<std::endl;
    }
    for(size_t i = 0; i <tempMat.m_row; i++){
      for(size_t j = 0; j <tempMat.m_col; j++){
        for(size_t k = 0; k < m_col; k++){
        tempMat.mat[i][j] += a.mat[i][k] * b.mat[k][j];
        }
      }
    }
    return tempMat;
  }
  
  T dot(const matrix<T>&b){
    T sum = 0; 
    if(m_col != b.m_col){
      std::cout<<"Invalid matrix sizes"<<std::endl;
    }
    for(size_t i = 0; i < m_row; i++ ){
      for(size_t j = 0; j <m_col; j++){
        sum += mat[i][j] * b.mat[i][j]; 
      }
    }
    return sum;
  }
  
  matrix block(size_t i,size_t j,size_t p,size_t q){
     if (i + p > this->m_row || j + q > this->m_col) {
        throw std::out_of_range("Block indices out of range");
    }
    matrix<T> tempMat(p, q);
    for(size_t a = 0; a < p; ++a){
      for(size_t b = 0; b < q; ++b){
        tempMat.mat[a][b] = mat[a+i][b+j]; 
      }
    }
    return tempMat;
  }
 
  void TP(){
    matrix<T> tempMat(this->m_row, this->m_col); 
    for(size_t i = 0; i < this->m_row; i++){
      for(size_t j = 0; j < this->m_col; j++){
        tempMat.mat[i][j] = this->mat[i][j]; 
      }
    }
    size_t temp = this->m_row; 
    this->m_row = this->m_col; 
    this->m_col = temp;
    for(size_t i = 0; i < this->m_row; i++){
      for(size_t j = 0; j < this->m_col; j++){
        this->mat[i][j] = tempMat.mat[j][i]; 
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

  matrix operator*(const matrix<T> &rhs)const{
    matrix<T> tempMat(m_row, rhs.m_col); 
    if(m_col != rhs.m_row){
      std::cout<<"Invalid matrix sizes"<<std::endl;
    }
    for(size_t i = 0; i <tempMat.m_row; i++){
      for(size_t j = 0; j <tempMat.m_col; j++){
        for(size_t k = 0; k < m_col; k++){
        tempMat.mat[i][j] += mat[i][k] * rhs.mat[k][j];
        }
      }
    }
    return tempMat; 
  }

  matrix& operator*=(const matrix<T> rhs)const{
    return *this = *this * rhs;
  }

  matrix operator+(const matrix<T> &rhs)const{
    matrix<T> tempMat(m_row, m_col); 
    if(m_row != rhs.m_row && m_col != rhs.m_col){
      std::cout<<"Invalid matrix sizes"<<std::endl; 
    }
    for(size_t i = 0; i < tempMat.m_row; i++){
      for(size_t j = 0; j < tempMat.m_col; j++){
        tempMat.mat[i][j] = mat[i][j]+rhs.mat[i][j];  
      }
    }
   return tempMat;  
  }

  matrix& operator+=(const matrix<T> rhs)const{
    return *this = *this + rhs; 
  }

  matrix operator-(const matrix<T> &rhs)const{
    matrix<T> tempMat(m_row, m_col); 
    if(m_row != rhs.m_row && m_col != rhs.m_col){
      std::cout<<"Invalid matrix sizes"<<std::endl; 
    }
    for(size_t i = 0; i < tempMat.m_row; i++){
      for(size_t j = 0; j < tempMat.m_col; j++){
        tempMat.mat[i][j] = mat[i][j]-rhs.mat[i][j];  
      }
    }
   return tempMat;  
  }

  matrix& operator -=(const matrix<T> rhs)const{
    return *this = *this - rhs; 
  }

 };

#pragma pack(pop)
}
#endif
