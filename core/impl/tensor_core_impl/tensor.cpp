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

__m512 tens::ops::fast_exp(__m512 input_vec){
  __m512  t; 
  __m512  f; 
  __m512  p; 
  __m512  r;
  __m512i i;
  __m512i j;

  const __m512 l2e = _mm512_set1_ps (1.442695041f);
  const __m512 l2h = _mm512_set1_ps (-6.93145752e-1f);
  const __m512 l2l = _mm512_set1_ps (-1.42860677e-6f);
  const __m512 c0 =  _mm512_set1_ps (0.041944388f);
  const __m512 c1 =  _mm512_set1_ps (0.168006673f);
  const __m512 c2 =  _mm512_set1_ps (0.499999940f);
  const __m512 c3 =  _mm512_set1_ps (0.999956906f);
  const __m512 c4 =  _mm512_set1_ps (0.999999642f);

  t = _mm512_mul_ps (input_vec, l2e);     
  r = _mm512_roundscale_ps (t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 

  f = _mm512_fmadd_ps (r, l2h, input_vec); 
  f = _mm512_fmadd_ps (r, l2l, f);

  i = _mm512_cvtps_epi32(t);     

  p = c0;                       
  p = _mm512_fmadd_ps (p, f, c1);
  p = _mm512_fmadd_ps (p, f, c2);
  p = _mm512_fmadd_ps (p, f, c3);
  p = _mm512_fmadd_ps (p, f, c4);

  j = _mm512_slli_epi32 (i, 23);
  r = _mm512_castsi512_ps (_mm512_add_epi32 (j, _mm512_castps_si512 (p)));

  return r;
}

tens::tensor tens::ops::add(const tens::tensor &left_tensor, const tens::tensor &right_tensor, memory::neural_arena &pool){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::add(&left_tensor, &right_tensor) :: cannot add two un-equal sized tensors"); 
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim         = left_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t)left_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i]    = left_tensor.shape.dims[i]; 
      output_tensor.shape.strides[i] = left_tensor.shape.strides[i];
    }
    output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel()); 

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

tens::tensor tens::ops::add(const tens::tensor &left_tensor, float scalar, memory::neural_arena &pool){
  if(left_tensor.shape.numel() <= 0){
    CPPDL_FATAL("tens::ops::add(&left_tensor, &right_tensor) :: cannot add a scalar to a 0 sized tensor"); 
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim         = left_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t)left_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i]    = left_tensor.shape.dims[i]; 
      output_tensor.shape.strides[i] = left_tensor.shape.strides[i];
    }
    output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());

    __m512 scalar_vec  = _mm512_set1_ps(scalar);    
    
    size_t tensor_size = left_tensor.shape.numel(); 
    size_t i           = 0; 
    float *left_data   = left_tensor.tensor_data; 
    float *out_data    = output_tensor.tensor_data; 
    
    for(; i + 15 < tensor_size; i += 16){
      __m512 left_vec  = _mm512_loadu_ps (&left_data[i]);
      __m512 add_vec   = _mm512_add_ps   (left_vec, scalar_vec); 
      _mm512_storeu_ps                   (&out_data[i], add_vec);
    }
    for(; i < tensor_size; ++i){
      output_tensor.tensor_data[i] = left_tensor.tensor_data[i] + scalar;
    }
    return output_tensor;
  }
}

tens::tensor tens::ops::sub(const tens::tensor &left_tensor, const tens::tensor &right_tensor, memory::neural_arena &pool){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::sub(&left_tensor, &right_tensor) :: cannot sub two un-equal sized tensors"); 
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim         = left_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t)left_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i]    = left_tensor.shape.dims[i]; 
      output_tensor.shape.strides[i] = left_tensor.shape.strides[i];
    } 
    output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());
   
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

tens::tensor tens::ops::mul(const tens::tensor &left_tensor, const tens::tensor &right_tensor, memory::neural_arena &pool){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::mul(&left_tensor, &right_tensor) :: cannot mul two un-equal sized tensors"); 
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim         = left_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t)left_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i]    = left_tensor.shape.dims[i]; 
      output_tensor.shape.strides[i] = left_tensor.shape.strides[i];
    } 
    output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());    
    
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

tens::tensor tens::ops::div(const tens::tensor &left_tensor, const tens::tensor &right_tensor, memory::neural_arena &pool){
  if(left_tensor.shape.numel() != right_tensor.shape.numel()){
    CPPDL_FATAL("tens::ops::div(&left_tensor, &right_tensor) :: cannot div two un-equal sized tensors"); 
    throw std::runtime_error(":: ABORTING ::");
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim         = left_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t)left_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i]    = left_tensor.shape.dims[i]; 
      output_tensor.shape.strides[i] = left_tensor.shape.strides[i];
    } 
    output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());   
    
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

tens::tensor tens::ops::scale(const tens::tensor &input_tensor, float scale, memory::neural_arena &pool){
  if(input_tensor.shape.numel() <= 0){
    CPPDL_FATAL("tens::ops::scale(&input_tensor, scale) :: cannot scale a 0 sized tensor :: returning original tensor and continuing"); 
    return input_tensor; 
  }else{
    tens::tensor output_tensor; 
    output_tensor.shape.ndim         = input_tensor.shape.ndim;
    for(size_t i = 0; i < (size_t)input_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i]    = input_tensor.shape.dims[i]; 
      output_tensor.shape.strides[i] = input_tensor.shape.strides[i];
    }
    output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());    
    
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

tens::tensor tens::ops::root(const tens::tensor &input_tensor, memory::neural_arena &pool){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::root(&input_tensor, scale) :: cannot get root of elems from a 0 sized tensor" 
         ); 
  tens::tensor output_tensor; 
  output_tensor.shape.ndim         = input_tensor.shape.ndim;
  for(size_t i = 0; i < (size_t)input_tensor.shape.ndim; ++i){
    output_tensor.shape.dims[i]    = input_tensor.shape.dims[i]; 
    output_tensor.shape.strides[i] = input_tensor.shape.strides[i];
  } 
  output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());
 
  size_t tensor_size = input_tensor.shape.numel(); 
  size_t i = 0; 
  for(; i + 15 < tensor_size; i += 16){
    __m512 input_vec = _mm512_loadu_ps(&input_tensor.tensor_data[i]);
    __m512 root_vec  = _mm512_sqrt_ps(input_vec); 
    _mm512_storeu_ps(&output_tensor.tensor_data[i], root_vec); 
  }
  for(; i < tensor_size; ++i){
    output_tensor.tensor_data[i] = std::sqrt(input_tensor.tensor_data[i]);
  }
  return output_tensor; 
}

tens::tensor tens::ops::tanh(const tens::tensor &input_tensor, memory::neural_arena &pool){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::tanh(&input_tensor) :: cannot apply tanh to a 0 sized tensor" 
         ); 
  tens::tensor output_tensor; 
  output_tensor.shape.ndim = input_tensor.shape.ndim;
  for(size_t i = 0; i < (size_t)input_tensor.shape.ndim; ++i){
    output_tensor.shape.dims[i] = input_tensor.shape.dims[i]; 
    output_tensor.shape.strides[i] = input_tensor.shape.strides[i];
  } 
  output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());  
  
  size_t tensor_size = input_tensor.shape.numel();
  
  __m512 pos_clamp = _mm512_set1_ps(10.0f);
  __m512 neg_clamp = _mm512_set1_ps(-10.0f);
  __m512 neg_vec   = _mm512_set1_ps(-1.0f); 
  
  size_t i = 0; 
  for(; i + 15 < tensor_size; i += 16){
    __m512 input_vec = _mm512_loadu_ps(&input_tensor.tensor_data[i]);
    
    input_vec = _mm512_min_ps(input_vec, pos_clamp);
    input_vec = _mm512_max_ps(input_vec, neg_clamp);
    
    __m512 neg_input_vec = _mm512_mul_ps(input_vec, neg_vec); 
    __m512 pos_exp_vec   = tens::ops::fast_exp(input_vec);
    __m512 neg_exp_vec   = tens::ops::fast_exp(neg_input_vec);
    __m512 top_diff      = _mm512_sub_ps(pos_exp_vec, neg_exp_vec); 
    __m512 bot_add       = _mm512_add_ps(pos_exp_vec, neg_exp_vec); 
    __m512 div_vec       = _mm512_div_ps(top_diff, bot_add); 
    _mm512_storeu_ps(&output_tensor.tensor_data[i], div_vec); 
  }
  for(; i < tensor_size; ++i){
    output_tensor.tensor_data[i] = std::tanh(input_tensor.tensor_data[i]); 
  }
  return output_tensor; 
}

tens::tensor tens::ops::var(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis, bool keep_dim){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::var(&input_tensor, scale) :: cannot get root of elems from a 0 sized tensor" 
         ); 
  assert(axis > 0 
         && "tens::ops::var(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::var(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
           ); 
  tens::tensor output_tensor; 
  output_tensor.shape.ndim         = input_tensor.shape.ndim;
  for(size_t i = 0; i < (size_t)input_tensor.shape.ndim; ++i){
    output_tensor.shape.dims[i]    = input_tensor.shape.dims[i]; 
    output_tensor.shape.strides[i] = input_tensor.shape.strides[i];
  }
  output_tensor.tensor_data        = input_tensor.tensor_data; 
  
  auto mean_tens  = tens::ops::mean (output_tensor, pool, axis, true);
  auto diff_tens  = tens::ops::sub  (input_tensor , mean_tens , pool);
  auto mul_tens   = tens::ops::mul  (diff_tens    , diff_tens , pool);
  auto final_tens = tens::ops::mean (mul_tens     , pool, axis, keep_dim); 
  
  return final_tens; 
}

tens::tensor tens::ops::sum(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
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
    for(int i = 0; i < input_tensor.shape.ndim; ++i){
      if(i != (int)axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(int i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size; 
  output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());  
  
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


tens::tensor tens::ops::mean(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
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
    for(int i = 0; i < input_tensor.shape.ndim; ++i){
      if(i != (int)axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(int i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());
  
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

tens::tensor tens::ops::max(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
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
    for(int i = 0; i < input_tensor.shape.ndim; ++i){
      if(i != (int)axis){
        output_tensor.shape.dims[j++] = input_tensor.shape.dims[i];  
      }
    }
  }

  if(output_tensor.shape.ndim > 0){
    output_tensor.shape.strides[input_tensor.shape.ndim - 1] = 1;
    for(int i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  output_tensor.tensor_data = pool.nn_alloc<float>(inner_size * outter_size);
  
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

tens::tensor tens::ops::min(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis, bool keep_dim){
  assert(input_tensor.shape.is_contiguous() 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: cannot sum tensor over non-contiguous tensor" 
         ); 
  assert(axis > 0 
         && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::sum(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
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
    for(int i = 0; i < input_tensor.shape.ndim; ++i){
      output_tensor.shape.dims[i] = (i == (int)axis) ? 1 : input_tensor.shape.dims[i]; 
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
    for(int i = output_tensor.shape.ndim - 2; i >= 0; --i){
      output_tensor.shape.strides[i] = output_tensor.shape.strides[i + 1] * output_tensor.shape.dims[i + 1]; 
    }
  }
  
  size_t output_numel = inner_size * outter_size;
  output_tensor.tensor_data = pool.nn_alloc<float>(inner_size * outter_size);
  
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

tens::tensor tens::ops::exp(const tens::tensor &input_tensor, memory::neural_arena &pool){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::exp(&input_tensor) :: cannot cannot apply exp to a 0 sized tensor" 
         ); 
  tens::tensor output_tensor; 
  output_tensor.shape.ndim         = input_tensor.shape.ndim;
  for(size_t i = 0; i < (size_t)input_tensor.shape.ndim; ++i){
    output_tensor.shape.dims[i]    = input_tensor.shape.dims[i]; 
    output_tensor.shape.strides[i] = input_tensor.shape.strides[i];
  }
  output_tensor.tensor_data = pool.nn_alloc<float>(output_tensor.shape.numel());      

  size_t i = 0; 
  for(; i + 15 < input_tensor.shape.numel(); i += 16){
    __m512 input_vec = _mm512_loadu_ps(&input_tensor.tensor_data[i]); 
    __m512 exp_vec   = tens::ops::fast_exp(input_vec);
    _mm512_storeu_ps(&output_tensor.tensor_data[i], exp_vec); 
  }
  for(; i < input_tensor.shape.numel(); ++i){
    output_tensor.tensor_data[i] = std::exp(input_tensor.tensor_data[i]); 
  }
  return output_tensor; 
}

tens::tensor tens::ops::layer_norm(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis, float epsilon, float gamma, float beta){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::layer_norm(&input_tensor, scale) :: cannot get root of elems from a 0 sized tensor" 
         ); 
  assert(axis > 0 
         && "tens::ops::layer_norm(&input_tensor, axis, keep_dim) :: selected axis does not exist :: cannot sum over axis less than 0 or 0" 
         );     
  assert(axis <= (size_t) input_tensor.shape.ndim 
           && "tens::ops::layer_norm(&input_tensor, axis, keep_dim) :: selected axis does not exist :: axis is greater than the tensors total dimension shape" 
           );

  auto diff_tens  = tens::ops::sub   (input_tensor, tens::ops::mean(input_tensor, pool, axis, true) , pool);
  auto comp_tens  = tens::ops::root  (tens::ops::add(tens::ops::var(input_tensor, pool, axis, true) , epsilon, pool), pool);
  auto div_tens   = tens::ops::div   (diff_tens , comp_tens, pool); 
  auto gamma_tens = tens::ops::scale (div_tens  , gamma    , pool); 
  auto final_tens = tens::ops::add   (gamma_tens, beta     , pool);

  return final_tens; 
}

tens::tensor tens::ops::gelu(const tens::tensor &input_tensor, memory::neural_arena &pool){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::gelu(&input_tensor, scale) :: cannot apply gelu() to a 0 sized tensor" 
         ); 

  auto half_tens            = tens::ops::scale (input_tensor, 0.5f             , pool);
  auto squared_tens         = tens::ops::mul   (input_tensor, input_tensor     , pool); 
  auto cubed_tens           = tens::ops::mul   (squared_tens, input_tensor     , pool); 
  auto scaled_cubed_tens    = tens::ops::scale (cubed_tens, 0.044715f          , pool); 
  auto complete_tens        = tens::ops::add   (scaled_cubed_tens, input_tensor, pool);
  auto scaled_complete_tens = tens::ops::scale (complete_tens, std::sqrt((2.0f / std::numbers::pi)), pool);
  auto tanh_tens            = tens::ops::tanh  (scaled_complete_tens, pool);
  auto shifted_tanh_tens    = tens::ops::add   (tanh_tens, 1.0f                , pool);
  auto final_tens           = tens::ops::mul   (half_tens, shifted_tanh_tens   , pool); 

  return final_tens; 
}

tens::tensor tens::ops::softmax(const tens::tensor &input_tensor, memory::neural_arena &pool, size_t axis){
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::gelu(&input_tensor, scale) :: cannot apply gelu() to a 0 sized tensor" 
         ); 

  auto max_tens      = tens::ops::max (input_tensor   , pool, axis, true);
  auto max_diff_tens = tens::ops::sub (input_tensor   , max_tens, pool);
  auto exp_tens      = tens::ops::exp (max_diff_tens  , pool);
  auto sum_tens      = tens::ops::sum (exp_tens, pool , axis, true);
  auto final_tens    = tens::ops::div (exp_tens       , sum_tens, pool);

  return final_tens; 
}

tens::tensor tens::ops::embedding(const tens::tensor &input_weights, const tens::tensor &input_indices, memory::neural_arena &pool){
  assert(input_weights.shape.numel() > 0 
         && "tens::ops::embedding(&input_weights, &input_indices) :: cannot apply embeddings to a 0 sized tensor" 
         ); 
 assert(input_indices.shape.numel() > 0 
         && "tens::ops::embedding(&input_weights, &input_indices) :: cannot apply embeddings to a 0 sized tensor" 
         ); 

  size_t embed_dim  = input_weights.shape.dims[1];
  size_t num_indices = input_indices.shape.numel(); 
  
  tens::tensor output_tensor; 
  output_tensor.shape.ndim = input_indices.shape.ndim + 1;

  for(int i = 0; i < input_indices.shape.ndim; ++i){
    output_tensor.shape.dims[i] = input_indices.shape.dims[i];   
  }
  output_tensor.shape.dims[output_tensor.shape.ndim - 1] = embed_dim;

  output_tensor.shape.strides[output_tensor.shape.ndim - 1] = 1;
  for(int i = output_tensor.shape.ndim - 2; i >= 0; --i){
    output_tensor.shape.strides[i] = input_indices.shape.strides[i + 1] * input_indices.shape.dims[i + 1]; 
  }
  output_tensor.tensor_data = pool.nn_alloc<float>(num_indices * embed_dim);
 
  for(size_t i = 0; i < num_indices; ++i){
    int token_id = (int)input_indices.tensor_data[i];
    float *mem_source = &input_weights.tensor_data[token_id * embed_dim];
    float *mem_dest   = &output_tensor.tensor_data[i * embed_dim];
    std::memcpy(mem_dest, mem_source, embed_dim * sizeof(float)); 
  }

  return output_tensor; 
}

tens::tensor tens::positional_encoders::sine_encoder(size_t sequence_length, size_t embed_dim, memory::neural_arena &pool){
  tens::tensor output_tensor; 
  output_tensor.shape.ndim = 2; 
  output_tensor.shape.dims[0] = sequence_length; 
  output_tensor.shape.dims[1] = embed_dim; 
  output_tensor.shape.strides[0] = embed_dim; 
  output_tensor.shape.strides[1] = 1; 
  output_tensor.tensor_data = pool.nn_alloc<float>(sequence_length * embed_dim);

  for(size_t pos = 0; pos < sequence_length; ++pos){
    for(size_t i = 0; i < embed_dim; i += 2){
      
      float div_term    = std::pow(10000.0f, (float)i / (float)embed_dim);
      float angle       = (float)pos / div_term; 
      
      size_t even_index = pos * embed_dim + 2 * i; 
      size_t odd_index  = pos * embed_dim + 2 * i + 1;

      output_tensor.tensor_data[even_index] = std::sin(angle); 
      output_tensor.tensor_data[odd_index]  = std::cos(angle); 
    }
  }
  return output_tensor; 
}

tens::tensor tens::ops::layer_norm(const tens::tensor &input_tensor, const tens::tensor &weight, const tens::tensor &bias, memory::neural_arena &pool, size_t axis, float epsilon) {
  assert(input_tensor.shape.numel() > 0 
         && "tens::ops::layer_norm(&input_tensor, &weight, &bias) :: cannot apply layer_norm to a 0 sized tensor" 
         ); 
  assert(weight.shape.numel() > 0 
         && "tens::ops::layer_norm(&input_tensor, &weight, &bias) :: cannot apply layer_norm to a 0 sized tensor" 
         ); 
  assert(bias.shape.numel() > 0 
         && "tens::ops::layer_norm(&input_tensor, &weight, &bias) :: cannot apply layer_norm to a 0 sized tensor" 
         ); 

  if (axis == (size_t)-1) {
    axis = input_tensor.shape.ndim - 1;
  }
  
  size_t outer_size = 1;
  size_t norm_size = input_tensor.shape.dims[axis];
  
  for (size_t i = 0; i < axis; i++) {
    outer_size *= input_tensor.shape.dims[i];
  }
  
  tens::tensor output;
  output.shape = input_tensor.shape;
  output.tensor_data = pool.nn_alloc<float>(input_tensor.shape.numel());
  
  for (size_t o = 0; o < outer_size; o++) {
    float* row_in = input_tensor.tensor_data + o * norm_size;
    float* row_out = output.tensor_data + o * norm_size;
    
    float mean = 0.0f;
    for (size_t n = 0; n < norm_size; n++) {
      mean += row_in[n];
    }
    mean /= (float)norm_size;
    
    float var = 0.0f;
    for (size_t n = 0; n < norm_size; n++) {
      float diff = row_in[n] - mean;
      var += diff * diff;
    }
    var /= (float)norm_size;
    
    float std_inv = 1.0f / std::sqrt(var + epsilon);
    
    for (size_t n = 0; n < norm_size; n++) {
      float normalized = (row_in[n] - mean) * std_inv;
      row_out[n] = normalized * weight.tensor_data[n] + bias.tensor_data[n];
    }
  }
  return output;
}

tens::tensor tens::ops::cpu_transpose_avx512(const tens::tensor& input, memory::neural_arena& pool) {
  size_t rows = input.shape.dims[0];
  size_t cols = input.shape.dims[1];

  tens::tensor out;
  out.shape.ndim = 2;
  out.shape.dims[0] = cols;
  out.shape.dims[1] = rows;
  out.shape.strides[0] = rows;
  out.shape.strides[1] = 1;
  out.tensor_data = pool.nn_alloc<float>(rows * cols);

  float* src = input.tensor_data;
  float* dst = out.tensor_data;

  const size_t BLOCK = 16;

  alignas(64) int indices[16];
  for (int i = 0; i < 16; ++i) indices[i] = i * rows;
  __m512i vindex = _mm512_load_si512(indices);

  size_t i = 0;
  for (; i + BLOCK <= rows; i += BLOCK) {
    size_t j = 0;
    for (; j + BLOCK <= cols; j += BLOCK) {
      for (size_t k = 0; k < BLOCK; ++k) {
        __m512 val = _mm512_loadu_ps(&src[(i + k) * cols + j]);

        void* addr = &dst[j * rows + (i + k)];
        _mm512_i32scatter_ps(addr, vindex, val, 4);
      }
    }
      
    for (; j < cols; ++j) {
      for (size_t k = 0; k < BLOCK; ++k) {
        dst[j * rows + (i + k)] = src[(i + k) * cols + j];
      }
    }
  }

  for (; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
  return out;
}
