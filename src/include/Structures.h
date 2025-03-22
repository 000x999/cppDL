#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#ifdef USE_VULKAN
  /*Placing this here since i'm going to be using a custom compute shader pipeline instead of CUDA or ROCm*/
#endif
#ifdef USE_OPENGL
  /*Going to use openGL in case some people use GPU's that don't suport vulkan*/
#endif
#ifdef USE_AVX256
  #include <immintrin.h>
  extern "C" int omp_get_thread_num();
  extern "C" int omp_get_num_threads();
  extern "C" int omp_set_dynamic(int x);
  extern "C" int omp_set_num_threads(int y);
  extern "C" int omp_get_max_threads();
#endif

#define MATRIX
namespace mat{
template <typename T>
struct matrix{
  size_t m_row; 
  size_t m_col;
  __attribute__((aligned(32))) std::vector<std::vector<T>> mat; 
  matrix(size_t row_in, size_t col_in)
        : 
    m_row(row_in), 
    m_col(col_in),
    mat(row_in, std::vector<T>(col_in, 0)) {}
};
  
template <typename T>
class MatOps{

private:
  int BLOCK_I = 8; 
  int BLOCK_J = 8; 
  int BLOCK_K = 8; 
   __attribute__((aligned(32))) mat::matrix<T> mat; 

public:
  MatOps(mat::matrix<T> &mat)
    :mat(mat){}

  void setBlockSize(const int size_in){
    BLOCK_I = size_in; 
    BLOCK_J = size_in; 
    BLOCK_K = size_in; 
  }

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
    
  T GetIndexVal(size_t i, size_t j){return this->mat.mat[j][j];}      
  size_t GetRow(){return this->mat.m_row;}
  size_t GetCol(){return this->mat.m_col;}

  void fillMat(){
    static std::random_device rd; 
    static std::mt19937 gen(rd()); 
    static std::uniform_real_distribution<float> randVal(1, 5);
    for(size_t i = 0; i < mat.m_row; i++){
      for(size_t j = 0; j < mat.m_col; j++){
      mat.mat[i][j] = randVal(gen);  
      }
    }
  }

  void zeros(){
    for(size_t i = 0; i < mat.m_row; i++){
      for(size_t j = 0; j < mat.m_col; j++){
      mat.mat[i][j] = 1;  
      }
    }
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
  
#if USE_AVX256
  void TP(){
    omp_set_dynamic(0); 
    omp_set_num_threads(omp_get_max_threads());
    matrix<T> tempMat(this->mat.m_row, this->mat.m_col); 
    #pragma omp parallel for
    for(size_t i = 0; i < this->mat.m_row; i++){
      for(size_t j = 0; j < this->mat.m_col; j++){
        tempMat.mat[i][j] = this->mat.mat[i][j]; 
      }
    }
    size_t temp = this->mat.m_row; 
    this->mat.m_row = this->mat.m_col; 
    this->mat.m_col = temp;
    #pragma omp parallel for
    for(size_t i = 0; i < this->mat.m_row; i++){
      for(size_t j = 0; j < this->mat.m_col; j++){
        this->mat.mat[i][j] = tempMat.mat[j][i]; 
      }
    }
    #pragma omp parallel
    printf("Thread %d out of %d\n", omp_get_thread_num(), omp_get_num_threads());
  }
#else
  void TP(){
    matrix<T> tempMat(this->mat.m_row, this->mat.m_col); 
    for(size_t i = 0; i < this->mat.m_row; i++){
      for(size_t j = 0; j < this->mat.m_col; j++){
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
#endif
  T sum(){
    T res = 0;
      /*
      T res = std::accumulate(mat.begin(), mat.end(),0.0, 
            [](matrix<T> lhs, const matrix<T>& rhs){
                              return std::accumulate(rhs.mat.begin(), lhs.mat.end(), lhs);
                              });*/
    typename std::vector<std::vector<T>>::iterator row; 
    typename std::vector<T>::iterator col; 
    for(row = mat.mat.begin(); row != mat.mat.end(); ++row){
      for(col = row->begin(); col != row->end(); ++col){
          res += *col; 
      }
    }
    return res; 
  }
  
#if USE_AVX256
  MatOps<T> operator*(const MatOps<T> &rhs)const{
  std::cout<<"DEBUG: AVX256_MATMUL_STARTED"<<std::endl; 
  #define N static_cast<int>(this->mat.m_row)
   __attribute__((aligned(32))) mat::matrix<float> A = this->mat;  
   __attribute__((aligned(32))) mat::matrix<float> B = rhs.mat;
   __attribute__((aligned(32))) mat::matrix<float> C(this->mat.m_row, this->mat.m_col);
   __attribute__((aligned(32))) mat::matrix<float> C_ref(this->mat.m_row, this->mat.m_col); 
   
    omp_set_dynamic(0); 
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for collapse(2)
      for(int index = 0; index < N; index += BLOCK_I) {
          for(int jindex = 0; jindex < N; jindex += BLOCK_J) {
              int iBlockSize = (index + BLOCK_I <= N) ? BLOCK_I : (N - index);
              int jBlockSize = (jindex + BLOCK_J <= N) ? BLOCK_J : (N - jindex);
              
              float cBlock[BLOCK_I][BLOCK_J] __attribute__((aligned(32)));

              for(int itile = 0; itile < iBlockSize; itile++) {
                  for(int jtile = 0; jtile < jBlockSize; jtile++) {
                      cBlock[itile][jtile] = 0.0f;
                  }
              }
              for(int kindex = 0; kindex < N; kindex += BLOCK_K) {
                  int kBlockSize = (kindex + BLOCK_K <= N) ? BLOCK_K : (N - kindex);

                  for(int itile = 0; itile < iBlockSize; itile++) {
                      int rowA = index + itile;
                      for(int jtile = 0; jtile < jBlockSize; jtile += 8) {
                          __m256 sumVec = _mm256_load_ps(&cBlock[itile][jtile]);

                          for(int ktile = 0; ktile < kBlockSize; ktile++) {
                              int kIndex = kindex + ktile;
                              __m256 aVal = _mm256_broadcast_ss(&A.mat[rowA][kIndex]);
                              __m256 bVal = _mm256_load_ps(&B.mat[kIndex][jindex + jtile]);

                              sumVec = _mm256_fmadd_ps(aVal, bVal, sumVec);
                          }
                          _mm256_store_ps(&cBlock[itile][jtile], sumVec);
                      }
                  }
              }
              for(int itile = 0; itile < iBlockSize; itile++) {
                  int rowC = index + itile;
                  for(int jtile = 0; jtile < jBlockSize; jtile++) {
                      C.mat[rowC][jindex + jtile] = cBlock[itile][jtile];
                  }
              }
          }
      }
    #pragma omp parallel
    printf("Thread %d out of %d\n", omp_get_thread_num(), omp_get_num_threads());
    mat::matrix<T> tempMat = C; 
    MatOps<T> ops(tempMat); 
    return ops; 
  }
#else
  /*Even though this only runs on the off chance someone doesn't support AVX256 in this day and age
   *I don't have to leave it O(2n^3), I'll implement the same cache aware blocking system I did for the AVX version later*/
  MatOps<T> operator*(const MatOps<T> &rhs) const{
    MatOps<T> tempMat = rhs;
    std::cout<<"DEBUG: std_matmul_started"<<std::endl;
    for(int i = 0; i < rhs.mat.m_row; ++i){
      for(int j = 0; j < rhs.mat.m_col; ++j){
        for(int k = 0; k < rhs.mat.m_col; ++k){
          tempMat.mat.mat[i][j] = rhs.mat.mat[i][k] * rhs.mat.mat[k][j]; 
        }
      }
    }
    return tempMat;
  }
#endif

  MatOps<T>& operator*=(const MatOps<T> &rhs)const{
    return *this = *this * rhs;
  }

  MatOps<T> operator+(const MatOps<T> &rhs)const{
    matrix<T> tempMat(mat.m_row, mat.m_col); 
    if(mat.m_row != rhs.mat.m_row && mat.m_col != rhs.mat.m_col){
      std::cout<<"Invalid matrix sizes"<<std::endl; 
    }
    for(size_t i = 0; i < tempMat.m_row; i++){
      for(size_t j = 0; j < tempMat.m_col; j++){
        tempMat.mat[i][j] = mat.mat[i][j]+rhs.mat.mat[i][j];  
      }
    }
    return tempMat;  
  }

  MatOps<T>& operator+=(const MatOps<T> rhs)const{
    return *this = *this + rhs; 
  }

  MatOps<T> operator-(const MatOps<T> &rhs)const{
    matrix<T> tempMat(mat.m_row, mat.m_col); 
    if(mat.m_row != rhs.mat.m_row && mat.m_col != rhs.mat.m_col){
      std::cout<<"Invalid matrix sizes"<<std::endl; 
    }
    for(size_t i = 0; i < tempMat.m_row; i++){
      for(size_t j = 0; j < tempMat.m_col; j++){
        tempMat.mat[i][j] = mat.mat[i][j]-rhs.mat.mat[i][j];  
      }
    }
    return tempMat;  
  }

  MatOps<T>& operator -=(const MatOps<T> rhs)const{
      return *this = *this - rhs; 
  }
};
};
#endif
