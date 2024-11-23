#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <cstdint>
#include <memory>
#include <vector>
#include <immintrin.h>

namespace Neuron{
  #define CONNECTION
  typedef struct Connection{
    float m_weight; 
    float m_deltaWeight; 
  }Connection;

  #define NEURON
  typedef struct Neuron{
    float m_outputVal;
    float m_grad;
    uint8_t m_index; 
    std::vector<std::shared_ptr<Connection>> m_OutPutWeights; 
  }Neuron;

  #define LAYER
  typedef struct Layer{
    std::vector<std::shared_ptr<Neuron>> Layers;    
  }Layer;
}

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
#endif
