#include "include/tensor_core/tensor.hpp"

size_t tens::tensor_shape::numel() const{
  if(ndim == 0){
      return 0; 
    }
    size_t accumulate = 1; 
    for(int i = 0; i < ndim; ++i){
      accumulate *= dims[i]; 
    }
    return accumulate; 
}

bool tens::tensor_shape::is_contiguous() const{
  size_t z_dim = 1; 
  for(int i = ndim - 1; i >= 0; --i){
    if(dims[i] != 1){
      if(strides[i] != z_dim){
        return false; 
      }
      z_dim *= dims[i]; 
    }
  }
  return true; 
}

tens::tensor tens::tensor::reshape(std::initializer_list<size_t> reshape_size) const{
  if(!shape.is_contiguous()){
    CPPDL_FATAL("tensor::reshape(std::initializer_list<size_t>) :: cannot reshape non-contiguous views"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::"); 
  }

  size_t new_total_size = 1; 
  for(auto dims : reshape_size){
    new_total_size *= dims; 
  }
  if(new_total_size != shape.numel()){
    CPPDL_FATAL("tensor::reshape(std::initializer_list<size_t>) :: reshape size mismatch"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::");
  }
  
  tensor new_view       = *this;
  new_view.shape.ndim   = 0;
  size_t running_stride = 1;  
  
  std::vector<size_t> temp_dims(reshape_size); 
  for(int i = temp_dims.size() - 1; i >= 0; --i){
    new_view.shape.dims   [i]    = temp_dims[i];
    new_view.shape.strides[i]    = running_stride; 
    running_stride              *= temp_dims[i]; 
  }
  new_view.shape.ndim = temp_dims.size();
  return new_view; 
}

tens::tensor tens::tensor::flatten() const { return reshape( { shape.numel() } ); }

tens::tensor tens::tensor::slice(size_t slice_index) const{
 if(slice_index >= shape.dims[0]){
    CPPDL_WARN("tensor::slice(size_t_) :: slice index out of bounds"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::out_of_range(":: ABORTING ::");
  }
  tensor new_view       = *this;
  new_view.tensor_data += slice_index * shape.strides[0];

  for(int i = 0; i < shape.ndim - 1; ++i){
    new_view.shape.dims   [i]    = shape.dims   [i + 1]; 
    new_view.shape.strides[i]    = shape.strides[i + 1]; 
  }
  new_view.shape.ndim--;
  return new_view; 
}

tens::tensor tens::tensor::transpose() const{
if(shape.ndim < 2){
    return *this; 
  }

  tensor new_view     = *this; 
  int last_dim        = shape.ndim - 1; 
  int second_last_dim = shape.ndim - 2;
  
  std::swap(new_view.shape.dims   [last_dim], new_view.shape.dims   [second_last_dim]); 
  std::swap(new_view.shape.strides[last_dim], new_view.shape.strides[second_last_dim]); 
  
  return new_view;
}

float* tens::tensor::begin() { return tensor_data; }
float* tens::tensor::end  () {
  if(!shape.is_contiguous()){
    CPPDL_FATAL("tensor::end() :: cannot reshape non-contiguous views"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::"); 
  }else{
    return tensor_data + shape.numel(); 
  }
}

void tens::tensor::randn(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  //std::memset(this->tensor_data, 0.0f, this->shape.numel() * sizeof(float)); 
  
  for(size_t i = 0; i < this->shape.numel(); ++i){
    this->tensor_data[i] = dist(gen);  
  }
}

