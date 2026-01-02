#include "include/tensor_core/tensor.hpp"
#include <process.h>

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

tens::tensor tens::ops::add(const tens::tensor &left_tensor, const tens::tensor &right_tensor){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::add(&left_tensor, &right_tensor) :: cannot add two un-equal sized tensors"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim       = left_tensor.shape.ndim;
    output_tensor.shape.dims[0]    = left_tensor.shape.dims[0]; 
    output_tensor.shape.dims[1]    = left_tensor.shape.dims[1];
    output_tensor.shape.strides[0] = left_tensor.shape.strides[0]; 
    output_tensor.shape.strides[1] = left_tensor.shape.strides[1];
    std::memset(output_tensor.tensor_data, 0.0f, output_tensor.shape.numel() * sizeof(float));
    
    size_t tensor_size = left_tensor.shape.numel(); 
    size_t i           = 0; 
    float *left_data   = left_tensor.tensor_data; 
    float *right_data  = right_tensor.tensor_data; 
    float *out_data    = output_tensor.tensor_data; 
    
    for(; i + 15 < tensor_size; i += 16){
      __m512 left_vec  = _mm512_loadu_ps (&left_data[i]);
      __m512 right_vec = _mm512_loadu_ps (&right_data[i]);
      __m512 add_vec   = _mm512_add_ps   (left_vec, right_vec); 
      _mm512_storeu_ps                   (&out_data[i], add_vec);
    }
    for(; i < tensor_size; ++i){
      output_tensor.tensor_data[i] = left_tensor.tensor_data[i] + right_tensor.tensor_data[i];
    }
    return output_tensor;
  }
}

tens::tensor tens::ops::sub(const tens::tensor &left_tensor, const tens::tensor &right_tensor){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::sub(&left_tensor, &right_tensor) :: cannot sub two un-equal sized tensors"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim       = left_tensor.shape.ndim;
    output_tensor.shape.dims[0]    = left_tensor.shape.dims[0]; 
    output_tensor.shape.dims[1]    = left_tensor.shape.dims[1];
    output_tensor.shape.strides[0] = left_tensor.shape.strides[0]; 
    output_tensor.shape.strides[1] = left_tensor.shape.strides[1];
    std::memset(output_tensor.tensor_data, 0.0f, output_tensor.shape.numel() * sizeof(float));
    
    size_t tensor_size = left_tensor.shape.numel(); 
    size_t i           = 0; 
    float *left_data   = left_tensor.tensor_data; 
    float *right_data  = right_tensor.tensor_data; 
    float *out_data    = output_tensor.tensor_data; 
    
    for(; i + 15 < tensor_size; i += 16){
      __m512 left_vec  = _mm512_loadu_ps (&left_data[i]);
      __m512 right_vec = _mm512_loadu_ps (&right_data[i]);
      __m512 sub_vec   = _mm512_sub_ps   (left_vec, right_vec); 
      _mm512_storeu_ps                   (&out_data[i], sub_vec);
    }
    for(; i < tensor_size; ++i){
      output_tensor.tensor_data[i] = left_tensor.tensor_data[i] - right_tensor.tensor_data[i];
    }
    return output_tensor;
  }
}

tens::tensor tens::ops::mul(const tens::tensor &left_tensor, const tens::tensor &right_tensor){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::mul(&left_tensor, &right_tensor) :: cannot mul two un-equal sized tensors"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim       = left_tensor.shape.ndim;
    output_tensor.shape.dims[0]    = left_tensor.shape.dims[0]; 
    output_tensor.shape.dims[1]    = left_tensor.shape.dims[1];
    output_tensor.shape.strides[0] = left_tensor.shape.strides[0]; 
    output_tensor.shape.strides[1] = left_tensor.shape.strides[1];
    std::memset(output_tensor.tensor_data, 0.0f, output_tensor.shape.numel() * sizeof(float));
    
    size_t tensor_size = left_tensor.shape.numel(); 
    size_t i           = 0; 
    float *left_data   = left_tensor.tensor_data; 
    float *right_data  = right_tensor.tensor_data; 
    float *out_data    = output_tensor.tensor_data; 
    
    for(; i + 15 < tensor_size; i += 16){
      __m512 left_vec  = _mm512_loadu_ps (&left_data[i]);
      __m512 right_vec = _mm512_loadu_ps (&right_data[i]);
      __m512 mul_vec   = _mm512_mul_ps   (left_vec, right_vec); 
      _mm512_storeu_ps                   (&out_data[i], mul_vec);
    }
    for(; i < tensor_size; ++i){
      output_tensor.tensor_data[i] = left_tensor.tensor_data[i] * right_tensor.tensor_data[i];
    }
    return output_tensor;
  }
}

tens::tensor tens::ops::div(const tens::tensor &left_tensor, const tens::tensor &right_tensor){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::div(&left_tensor, &right_tensor) :: cannot div two un-equal sized tensors"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim       = left_tensor.shape.ndim;
    output_tensor.shape.dims[0]    = left_tensor.shape.dims[0]; 
    output_tensor.shape.dims[1]    = left_tensor.shape.dims[1];
    output_tensor.shape.strides[0] = left_tensor.shape.strides[0]; 
    output_tensor.shape.strides[1] = left_tensor.shape.strides[1];
    std::memset(output_tensor.tensor_data, 0.0f, output_tensor.shape.numel() * sizeof(float));
    
    size_t tensor_size = left_tensor.shape.numel(); 
    size_t i           = 0; 
    float *left_data   = left_tensor.tensor_data; 
    float *right_data  = right_tensor.tensor_data; 
    float *out_data    = output_tensor.tensor_data; 
    
    for(; i + 15 < tensor_size; i += 16){
      __m512 left_vec  = _mm512_loadu_ps (&left_data[i]);
      __m512 right_vec = _mm512_loadu_ps (&right_data[i]);
      __m512 div_vec   = _mm512_div_ps   (left_vec, right_vec); 
      _mm512_storeu_ps                   (&out_data[i], div_vec);
    }
    for(; i < tensor_size; ++i){
      output_tensor.tensor_data[i] = left_tensor.tensor_data[i] / right_tensor.tensor_data[i];
    }
    return output_tensor;
  }
}

tens::tensor tens::ops::scale(const tens::tensor &input_tensor, size_t scale){
  if(input_tensor.shape.numel() <= 0){
    CPPDL_FATAL("tens::ops::scale(&input_tensor, scale) :: cannot scale a 0 sized tensor :: returning original tensor and continuing"); 
    printf("File: %s :: Line: %d", __FILE__, __LINE__);
    return input_tensor; 
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim       = input_tensor.shape.ndim;
    output_tensor.shape.dims[0]    = input_tensor.shape.dims[0]; 
    output_tensor.shape.dims[1]    = input_tensor.shape.dims[1];
    output_tensor.shape.strides[0] = input_tensor.shape.strides[0]; 
    output_tensor.shape.strides[1] = input_tensor.shape.strides[1];
    std::memset(output_tensor.tensor_data, 0.0f, output_tensor.shape.numel() * sizeof(float));
    
    size_t tensor_size = input_tensor.shape.numel(); 
    size_t i           = 0; 
    float *input_data  = input_tensor.tensor_data;
    float *out_data    = output_tensor.tensor_data; 
    __m512 scale_vec   = _mm512_set1_ps((float)scale);  
    
    for(; i + 15 < tensor_size; i += 16){
      __m512 left_vec  = _mm512_loadu_ps (&input_data[i]);
      __m512 mul_vec   = _mm512_mul_ps   (left_vec, scale_vec); 
      _mm512_storeu_ps                   (&out_data[i], mul_vec);
    }
    for(; i < tensor_size; ++i){
      output_tensor.tensor_data[i] = input_tensor.tensor_data[i] * scale;
    }
    return output_tensor;
  }
}

tens::tensor tens::ops::sum(const tens::tensor &input_tensor, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
           && printf("File: %s :: Line: %d", __FILE__, __LINE__)
           ); 
  
  size_t outter_size = 1; 
  size_t reduce_size = input_tensor.shape.dims[axis]; 
  size_t inner_size = 1; 

  for(size_t i = 0; i < axis; ++i){
    outter_size *= input_tensor.shape.dims[i];
  }

  for(size_t i = axis + 1; i < (size_t) input_tensor.shape.ndim; ++i){
    inner_size *= input_tensor.shape.dims[i];
  }

  tens::tensor output_tensor;
  
  if(keep_dim){
    output_tensor.shape.ndim = input_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i] = (i == axis) ? 1 : input_tensor.shape.dims[i]; 
    }
  }else{
    output_tensor.shape.ndim = input_tensor.shape.ndim - 1; 
    size_t j = 0; 
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      if(i != axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(size_t i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  std::memset(output_tensor.tensor_data, 0.0f, output_numel * sizeof(float));

  for(size_t i = 0; i < outter_size; ++i){
    for(size_t j = 0; j < inner_size; ++j){
      float acc = 0.0f; 
      
      for(size_t reduce = 0; reduce < reduce_size; ++reduce){
        size_t index = i * (inner_size * reduce_size) + reduce * inner_size + j;
        acc += input_tensor.tensor_data[index]; 
      }
      output_tensor.tensor_data[i * inner_size + j] = acc; 
    }
  }
  return output_tensor;
} 


tens::tensor tens::ops::mean(const tens::tensor &input_tensor, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
           && printf("File: %s :: Line: %d", __FILE__, __LINE__)
           ); 
  
  size_t outter_size = 1; 
  size_t reduce_size = input_tensor.shape.dims[axis]; 
  size_t inner_size = 1; 

  for(size_t i = 0; i < axis; ++i){
    outter_size *= input_tensor.shape.dims[i];
  }

  for(size_t i = axis + 1; i < (size_t) input_tensor.shape.ndim; ++i){
    inner_size *= input_tensor.shape.dims[i];
  }

  tens::tensor output_tensor;
  
  if(keep_dim){
    output_tensor.shape.ndim = input_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i] = (i == axis) ? 1 : input_tensor.shape.dims[i]; 
    }
  }else{
    output_tensor.shape.ndim = input_tensor.shape.ndim - 1; 
    size_t j = 0; 
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      if(i != axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(size_t i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  std::memset(output_tensor.tensor_data, 0.0f, output_numel * sizeof(float));

  for(size_t i = 0; i < outter_size; ++i){
    for(size_t j = 0; j < inner_size; ++j){
      float acc = 0.0f; 
      
      for(size_t reduce = 0; reduce < reduce_size; ++reduce){
        size_t index = i * (inner_size * reduce_size) + reduce * inner_size + j;
        acc += input_tensor.tensor_data[index]; 
      }
      output_tensor.tensor_data[i * inner_size + j] = acc / (float) reduce_size; 
    }
  }
  return output_tensor;
} 

tens::tensor tens::ops::max(const tens::tensor &input_tensor, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
           && printf("File: %s :: Line: %d", __FILE__, __LINE__)
           ); 
  
  size_t outter_size = 1; 
  size_t reduce_size = input_tensor.shape.dims[axis]; 
  size_t inner_size = 1; 

  for(size_t i = 0; i < axis; ++i){
    outter_size *= input_tensor.shape.dims[i];
  }

  for(size_t i = axis + 1; i < (size_t) input_tensor.shape.ndim; ++i){
    inner_size *= input_tensor.shape.dims[i];
  }

  tens::tensor output_tensor;
  
  if(keep_dim){
    output_tensor.shape.ndim = input_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i] = (i == axis) ? 1 : input_tensor.shape.dims[i]; 
    }
  }else{
    output_tensor.shape.ndim = input_tensor.shape.ndim - 1; 
    size_t j = 0; 
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      if(i != axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(size_t i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  std::memset(output_tensor.tensor_data, 0.0f, output_numel * sizeof(float));

  for(size_t i = 0; i < outter_size; ++i){
    for(size_t j = 0; j < inner_size; ++j){
      float curr_max = -std::numeric_limits<float>::infinity(); 
      
      for(size_t reduce = 0; reduce < reduce_size; ++reduce){
        size_t index = i * (inner_size * reduce_size) + reduce * inner_size + j;
        curr_max = std::max(input_tensor.tensor_data[index], curr_max); 
      }
      output_tensor.tensor_data[i * inner_size + j] = curr_max; 
    }
  }
  return output_tensor;
}

tens::tensor tens::ops::min(const tens::tensor &input_tensor, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         && printf("File: %s :: Line: %d", __FILE__, __LINE__)
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
           && printf("File: %s :: Line: %d", __FILE__, __LINE__)
           ); 
  
  size_t outter_size = 1; 
  size_t reduce_size = input_tensor.shape.dims[axis]; 
  size_t inner_size = 1; 

  for(size_t i = 0; i < axis; ++i){
    outter_size *= input_tensor.shape.dims[i];
  }

  for(size_t i = axis + 1; i < (size_t) input_tensor.shape.ndim; ++i){
    inner_size *= input_tensor.shape.dims[i];
  }

  tens::tensor output_tensor;
  
  if(keep_dim){
    output_tensor.shape.ndim = input_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i] = (i == axis) ? 1 : input_tensor.shape.dims[i]; 
    }
  }else{
    output_tensor.shape.ndim = input_tensor.shape.ndim - 1; 
    size_t j = 0; 
    for(size_t i = 0; i < (size_t) input_tensor.shape.ndim; ++i){
      if(i != axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(size_t i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  std::memset(output_tensor.tensor_data, 0.0f, output_numel * sizeof(float));

  for(size_t i = 0; i < outter_size; ++i){
    for(size_t j = 0; j < inner_size; ++j){
      float curr_min = std::numeric_limits<float>::infinity(); 
      
      for(size_t reduce = 0; reduce < reduce_size; ++reduce){
        size_t index = i * (inner_size * reduce_size) + reduce * inner_size + j;
        curr_min = std::min(input_tensor.tensor_data[index], curr_min); 
      }
      output_tensor.tensor_data[i * inner_size + j] = curr_min; 
    }
  }
  return output_tensor;
}
