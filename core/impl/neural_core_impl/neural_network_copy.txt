#include "../../include/neural_core/neural_network.hpp"
#include <process.h>

neural::layer_data::layer_data(size_t input, size_t output)
  : layer_input_size(input),
    layer_output_size(output),
    weights(input * output),
    biases(output),
    input(input,0.0f),
    output(output,0.0f),
    layer_deriv_in(input,0.0f),
    layer_deriv_out(output,0.0f),
    weight_grad(input*output, 0.0f),
    bias_grad(output,0.0f), 
    weights_fp16(input * output){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto start = neural::nn::nanos();  
  #if USE_AVX256
    std::cout << "AVX FILL START: " << '\n';
    float *bias_data   = biases.data(); 
    float *weight_data = weights.data(); 
    size_t i = 0; 
    
    for(; i + 7 < weights.size(); i += 8){
      alignas(32) float temp[8]; 
      
      for(int k = 0; k < 8; ++k){
        temp[k] = dist(gen); 
      }
      __m256 random_vec = _mm256_load_ps(temp);
      _mm256_storeu_ps(&weight_data[i], random_vec);
    }
    for(; i < weights.size(); ++i){
      weight_data[i] = dist(gen);
    }
    
    size_t j = 0; 
    for(; j + 7 < biases.size(); j += 8){
      alignas(32) float temp[8]; 
      
      for(int k = 0; k < 8; ++k){
        temp[k] = dist(gen); 
      }
      __m256 random_vec = _mm256_load_ps(temp);
      _mm256_storeu_ps(&bias_data[j], random_vec);
    }
    for(; j < biases.size(); ++j){
      bias_data[j] = dist(gen);
    }
  #else
    std::cout << "NON AVX FILL: " << '\n'; 
    for (auto &w : weights) {
      w = dist(gen);
    }
    for (auto &b : biases) {
      b = dist(gen);
    }
  #endif
    auto end = neural::nn::nanos(); 
    std::cout << "Network fill time = " << (end - start) * 1e-9 << "s" << '\n'; 
  }

#if USE_BLAS
#if USE_AVX256
void neural::linear::forward_batched(const std::vector<float> &input_batch, size_t batch_size, neural::layer_data &data, std::vector<float> &output_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size;

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATAL("LINEAR::FORWARD_BATCHED INPUT SIZE MISMATCH :: ASSERTION FAILED");
  }
  assert(input_batch.size() == input_size * batch_sie); 
#endif

  data.input = input_batch; 
  if(output_batch.size() != output_size * batch_size){
    output_batch.assign(output_size * batch_size, 0.0f); 
  }else{
    std::fill(output_batch.begin(), output_batch.end(), 0.0f); 
  }
  
  data.output = output_batch; 
  level3::mat_ops_view mat_w { output_size, input_size, input_size, data.weights.data()                    }; 
  level3::mat_ops_view mat_x { input_size , batch_size, batch_size, const_cast<float*>(input_batch.data()) };
  level3::mat_ops_view mat_y { output_size, batch_size, batch_size, output_batch.data()                    };
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, mat_w, mat_x, 1.0f, 0.0f, mat_y);

  for(size_t i = 0; i < output_size; ++i){
    float bias_data = data.biases[i]; 
    float *mat_y_row_data = &output_batch[i * batch_size]; 
    for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
      mat_y_row_data[batch_index] += bias_data; 
    }
  }
  data.output = output_batch; 
}
#endif 
#else 
void neural::linear::forward_batched(const std::vector<float> &input_batch, size_t batch_size, neural::layer_data &data, std::vector<float> &output_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size; 

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATAL("LINEAR::FORWARD_BATCHED INPUT SIZE MISMATCH :: ASSERTION FAILED");
  }
  assert(input_batch.size == input_size * batch_sie); 
#endif
  data.input = input_batch;
  output_batch.assign(output_size * batch_size, 0.0f);
  level3::mat_ops_view mat_w { output_size,  input_size,  input_size, data.weights.data()                    };
  level3::mat_ops_view mat_x { input_size ,  batch_size,  batch_size, const_cast<float*>(input_batch.data()) }; 
  level3::mat_ops_view mat_y { output_size,  batch_size,  batch_size, output_batch.data()                    }; 

  level3::blas::gemm(output_size, input_size, batch_size, mat_w, mat_x, 1.0f, 0.0f, mat_y); 
  
  for(size_t i = 0; i < output_size; ++i){
    float batch_biases = data.biases[i]; 
    for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
      output_batch[ i * batch_size + batch_index ] += batch_biases; 
    }
  }
  data.output = output_batch; 
} 
#endif 

#if USE_AVX256
void neural::linear::forward(const std::vector<float> &layer_input_activations, neural::layer_data &data){
#if DEBUG
  if(layer_input_activations.size() != data.layer_input_size){
    CPPDL_FATAL("layer activations not equal to data layer input size :: ASSERTION FAILED"); 
  }
  assert(layer_input_activations.size() == data.layer_input_size); 
#endif
  data.input                 = layer_input_activations;
  const size_t output_size   = data.layer_output_size; 
  const size_t input_size    = data.layer_input_size;
  for(size_t i = 0; i < output_size; ++i){
    const float *weight_rows = &data.weights[i * input_size];
    __m256 sum_vec           = _mm256_setzero_ps(); 
   
    size_t j = 0; 
    for(;j + 7 < input_size; j += 8){
      __m256 weights_vec     = _mm256_loadu_ps(&weight_rows[j]); 
      __m256 activations_vec = _mm256_loadu_ps(&layer_input_activations[j]);
      sum_vec                = _mm256_fmadd_ps(weights_vec, activations_vec, sum_vec); 
    }

    alignas(32) float temp_vec[8]; 
    _mm256_store_ps(temp_vec, sum_vec); 
    float sum = data.biases[i] + temp_vec[0] + temp_vec[1] + temp_vec[2] 
                + temp_vec[3]  + temp_vec[4] + temp_vec[5] + temp_vec[6] + temp_vec[7];
    for(;j < input_size; ++j){
      sum += weight_rows[j] * layer_input_activations[j]; 
    }
    data.output[i] = sum; 
  }
}
#else
void neural::linear::forward(const std::vector<float> &layer_input_activations, neural::layer_data &data){
#if DEBUG
  if(layer_input_activations.size() != data.layer_input_size){
    CPPDL_FATAL("layer activations not equal to data layer input size :: ASSERTION FAILED"); 
  }
  assert(layer_input_activations.size() == data.layer_input_size); 
#endif
  data.input = layer_input_activations; 
  for(size_t i = 0; i < data.layer_output_size; ++i){
    float sum = data.biases[i]; 
    for(size_t j = 0; j < data.layer_input_size; ++j){
      sum += data.weights[i * data.layer_input_size + j] * layer_input_activations[j]; 
    }
    data.output[i] = sum; 
  }
}
#endif

void neural::linear::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, neural::layer_data &data, std::vector<float> &grad_input_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size;

#if DEBUG
  if(grad_output_batch.size() != output_size * batch_size){
    CPPDL_FATAL("LINEAR::BACKWARDS_BATCHED() :: grad_output_batch size mismatch :: ASSERTION FAILED");
    assert(grad_output_batch.size() == output_size * batch_size);
  }

  if(data.input.size() != input_size * batch_size){
    CPPDL_FATAL("LINEAR::BACKWARDS_BATCHED() :: data.input size mismatch :: ASSERTION FAILED");
    assert(data.input.size() == input_size * batch_size);
  }
#endif
  level3::mat_ops_view dy_grad_view { output_size, batch_size, batch_size, const_cast<float*>(grad_output_batch.data()) };
  level3::mat_ops_view x_grad_view  { input_size , batch_size, batch_size, data.input.data()                            };
  level3::mat_ops_view w_grad_view  { output_size, input_size, input_size, data.weights.data()                          };

  if(data.weight_grad.size() != output_size * input_size){
    data.weight_grad.assign(output_size * input_size, 0.0f); 
  }else{
    std::fill(data.weight_grad.begin(), data.weight_grad.end(), 0.0f); 
  }

  level3::mat_ops_view dw_grad_view { output_size, input_size, input_size, data.weight_grad.data() };
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::transpose, dy_grad_view, x_grad_view, 1.0f, 0.0f, dw_grad_view);

  if(grad_input_batch.size() != input_size * batch_size){
    grad_input_batch.assign(input_size * batch_size, 0.0f); 
  }else{
    std::fill(grad_input_batch.begin(), grad_input_batch.end(), 0.0f);
  }

  level3::mat_ops_view dx_grad_view { input_size, batch_size, batch_size, grad_input_batch.data() };
  level3::blas::crush_gemm(level3::transpose_gemm::transpose, level3::transpose_gemm::no_transpose, w_grad_view, dy_grad_view, 1.0f, 0.0f, dx_grad_view);

  data.layer_deriv_in = grad_input_batch; 

  if(data.bias_grad.size() != output_size){
    data.bias_grad.assign(output_size, 0.0f); 
  }else{
    std::fill(data.bias_grad.begin(), data.bias_grad.end(), 0.0f);
  }

  for(size_t i = 0; i < output_size; ++i){
    const float *dy_grad_data = &grad_output_batch[i * batch_size]; 
    float eno_sum             = 0.0f; 
    for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
      eno_sum += dy_grad_data[batch_index]; 
    }
    data.bias_grad[i] = eno_sum; 
  }
}

#if USE_AVX256
void neural::linear::backwards(const std::vector<float> &layer_deriv_out, neural::layer_data &data){
#if DEBUG
  if(layer_deriv_out.size() != data.layer_output_size){
    CPPDL_FATAL("layer derivative output size is not equal to data layer ouput size :: ASSERTION FAILED"); 
  }
  assert(layer_deriv_out.size() == data.layer_output_size); 
#endif

  std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f); 
  std::fill(data.weight_grad.begin(), data.weight_grad.end(), 0.0f); 
  std::fill(data.bias_grad.begin(), data.bias_grad.end(), 0.0f);  
  
  const size_t output_size = data.layer_output_size; 
  const size_t input_size  = data.layer_input_size; 
 
  for(size_t i = 0; i < output_size; ++i){
    const float lane_deriv               = layer_deriv_out[i];
    data.bias_grad[i]                    = lane_deriv;
    const float *weight_rows             = &data.weights[i * input_size];
    float       *weight_grad_rows        = &data.weight_grad[i * input_size];

    __m256 deriv_in_vec = _mm256_set1_ps(lane_deriv); 
    size_t j = 0; 
    for(; j + 7 < input_size; j += 8){
      __m256 weights_vec     = _mm256_loadu_ps(&weight_rows[j]);
      __m256 old_deriv_in    = _mm256_loadu_ps(&data.layer_deriv_in[i]);
      __m256 new_deriv_vec   = _mm256_fmadd_ps(weights_vec, deriv_in_vec, old_deriv_in);
      _mm256_storeu_ps(&data.layer_deriv_in[j], new_deriv_vec);
      
      __m256 input_vec       = _mm256_loadu_ps(&data.input[j]);
      __m256 deriv_out_vec   = _mm256_mul_ps(deriv_in_vec, input_vec);
      _mm256_storeu_ps(&weight_grad_rows[j], deriv_out_vec);
    }
    for(; j < input_size; ++j){
      data.layer_deriv_in[j] += weight_rows[j] * lane_deriv;
      weight_grad_rows[j]     = lane_deriv * data.input[j];  
    }
  }
  data.layer_deriv_out = layer_deriv_out; 
}
#else
void neural::linear::backwards(const std::vector<float> &layer_deriv_out, neural::layer_data &data){
#if DEBUG
  if(layer_deriv_out.size() != data.layer_output_size){
    CPPDL_FATAL("layer derivative output size is not equal to data layer ouput size :: ASSERTION FAILED"); 
  }
  assert(layer_deriv_out.size() == data.layer_output_size); 
#endif
  std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f); 
  std::fill(data.weight_grad.begin(), data.weight_grad.end(), 0.0f); 
  std::fill(data.bias_grad.begin(), data.bias_grad.end(), 0.0f);  
  for(size_t i = 0; i < data.layer_output_size; ++i){
    for(size_t j = 0; j < data.layer_input_size; ++j){
      data.layer_deriv_in[j] += data.weights[i * data.layer_input_size +j] * layer_deriv_out[i];
    }
  }
  for(size_t i = 0; i < data.layer_output_size; ++i){
    data.bias_grad[i] = layer_deriv_out[i]; 
    for(size_t j = 0; j < data.layer_input_size; ++j){
      data.weight_grad[i * data.layer_input_size + j] = layer_deriv_out[i] * data.input[j];
    }
  }
  data.layer_deriv_out = layer_deriv_out; 
}
#endif

#if USE_AVX256
void neural::linear::update(neural::layer_data &data, float learning_rate){
  const size_t output_size = data.layer_output_size; 
  const size_t input_size  = data.layer_input_size;
  const float  lane_learning_rate = learning_rate;
  __m256 learn_vec = _mm256_set1_ps(lane_learning_rate); 

  for(size_t i = 0; i < output_size; ++i){
    float       *weight_rows            = &data.weights[i * input_size];
    const float *weight_grad_rows       = &data.weight_grad[i * input_size]; 
    
    data.biases[i] -= learning_rate * data.bias_grad[i];
    
    size_t j = 0; 
    for(; j + 7 < input_size; j += 8){
      __m256 old_vec     = _mm256_loadu_ps(&weight_rows[j]); 
      __m256 grad_vec    = _mm256_loadu_ps(&weight_grad_rows[j]); 
      __m256 mulled_vec  = _mm256_mul_ps(learn_vec, grad_vec);
      __m256 updated_vec = _mm256_sub_ps(old_vec, mulled_vec);
      _mm256_storeu_ps(&weight_rows[j], updated_vec);
    }
    for(; j < input_size; ++j){
      weight_rows[j] -= learning_rate * weight_grad_rows[j];      
    }   
  }
}
#else
void neural::linear::update(neural::layer_data &data, float eta){
  for(size_t i = 0; i < data.layer_output_size; ++i){
    data.biases[i] -= eta * data.bias_grad[i]; 
    for(size_t j = 0; j < data.layer_input_size; ++j){
      data.weights[i * data.layer_input_size +j] -= eta * data.weight_grad[i * data.layer_input_size + j]; 
    }
  }
}
#endif

void neural::linear_relu_fused::forward_batched(const std::vector<float> &input_batch, size_t batch_size, layer_data &data, std::vector<float> &output_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size;

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATAL("linear_relu_fused::forward_batched() :: input size mismatch :: ASSERTION FAILED"); 
  }
  assert(input_batch.size() == input_size * batch_size); 
#endif 

  data.input = input_batch; 
  if(output_batch.size() != output_size * batch_size){
    output_batch.assign(output_size * batch_size, 0.0f);
  }else{
    std::fill(output_batch.begin(), output_batch.end(), 0.0f); 
  }

  level3::mat_ops_view mat_w { output_size, input_size, input_size, data.weights.data()                    };
  level3::mat_ops_view mat_x { input_size , batch_size, batch_size, const_cast<float*>(input_batch.data()) };
  level3::mat_ops_view mat_y { output_size, batch_size, batch_size, output_batch.data()                    };
  
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, mat_w, mat_x, 1.0f, 0.0f, mat_y); 
  
  float *output_batch_rows = output_batch.data();
  __m256 zero_vec         = _mm256_setzero_ps(); 
  
  for(size_t output_index = 0; output_index < output_size; ++output_index){
    const float biases    = data.biases[output_index]; 
    const __m256 bias_vec = _mm256_set1_ps(biases); 
    float *bias_row_data  = output_batch_rows + output_index * batch_size;

    size_t j = 0; 
    for(; j + 7 < batch_size; j += 8){
      __m256 row_vec      = _mm256_loadu_ps(&bias_row_data[j]);
      row_vec             = _mm256_add_ps(row_vec, bias_vec); 
      __m256 relu_vec     = _mm256_max_ps(row_vec, zero_vec); 
      _mm256_storeu_ps(&bias_row_data[j], relu_vec); 
    }
    for(;j < batch_size; ++j){
      float curr_relu = bias_row_data[j] + biases;
      bias_row_data[j] = (curr_relu > 0.0f) ? curr_relu : 0.0f; 
    }
  }
  data.output = output_batch;
}

void neural::linear_relu_fused::forward(const std::vector<float> &act, layer_data &data){}
void neural::linear_relu_fused::backwards(const std::vector<float> &act, layer_data &data){}
void neural::linear_sigmoid_fused::forward(const std::vector<float> &act, layer_data &data){}
void neural::linear_sigmoid_fused::backwards(const std::vector<float> &act, layer_data &data){}


void neural::linear_relu_fused::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size; 
  const size_t total_size  = output_size * batch_size; 

#if DEBUG 
  if(grad_output_batch.size() != total_size || data.output.size() != total_size || data.input.size() != input_size * batch_size){
    CPPDL_FATAL("neural::linear_relu_fused::backwards_batched() :: size mismatch :: ASSERTION FAILED"); 
  }
  assert(grad_output_batch.size() == total_size && data.output.size() == total_size && data.input.size() == input_size * batch_size);
#endif

  std::vector<float> function_grad(total_size);
  float *function_grad_data = function_grad.data();
  const __m256 zero_vec   = _mm256_setzero_ps();
 
  size_t i = 0; 

  for(; i + 7 < total_size; i += 8){
    __m256 curr_data_vec   = _mm256_loadu_ps(&data.output[i]);
    __m256 d_data_vec      = _mm256_loadu_ps(&grad_output_batch[i]);
    __m256 mask_vec        = _mm256_cmp_ps  (curr_data_vec, zero_vec, _CMP_GT_OQ);
    __m256 deriv_vec       = _mm256_and_ps  (d_data_vec, mask_vec); 
    _mm256_storeu_ps                        (&function_grad_data[i], deriv_vec); 
  }
  for(; i < total_size; ++i){
    float curr_data        = data.output[i];
    float d_data           = grad_output_batch[i]; 
    float deriv_out        = (curr_data > 0.0f) ? d_data : 0.0f; 
    function_grad_data[i]  = deriv_out; 
  }
  data.layer_deriv_out = function_grad;

  if(data.weight_grad.size() != output_size * input_size){
    data.weight_grad.resize(output_size * input_size, 0.0f); 
  }else{
    std::fill(data.weight_grad.begin(), data.weight_grad.end(), 0.0f); 
  }
  
  level3::mat_ops_view function_grad_view { output_size, batch_size, batch_size, function_grad.data()    };
  level3::mat_ops_view mat_x_view         { input_size , batch_size, batch_size, data.input.data()       };
  level3::mat_ops_view dw_view            { output_size, input_size, input_size, data.weight_grad.data() };

  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::transpose, function_grad_view, mat_x_view, 1.0f, 0.0f, dw_view);

  if(grad_input_batch.size() != batch_size * input_size){
    grad_input_batch.resize(batch_size * input_size, 0.0f); 
  }else{
    std::fill(grad_input_batch.begin(), grad_input_batch.end(), 0.0f); 
  }

  level3::mat_ops_view mat_w_view { output_size, input_size, input_size, data.weights.data()             };
  level3::mat_ops_view dx_view    { input_size , batch_size, batch_size, grad_input_batch.data()         };

  level3::blas::crush_gemm(level3::transpose_gemm::transpose, level3::transpose_gemm::no_transpose, mat_w_view, function_grad_view, 1.0f, 0.0f, dx_view);

  data.layer_deriv_in = grad_input_batch; 

  if(data.bias_grad.size() != output_size){
    data.bias_grad.assign(output_size, 0.0f); 
  }else{
    std::fill(data.bias_grad.begin(), data.bias_grad.end(), 0.0f); 
  }

  for(size_t i = 0; i < output_size; ++i){
    const float *row_grad_data = &function_grad[i * batch_size]; 
    float curr_sum = 0.0f; 

    for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
      curr_sum += row_grad_data[batch_index]; 
    }
    data.bias_grad[i] = curr_sum;
  }
}

void neural::linear_relu_fused::update(layer_data &data, float eta){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size; 
  const size_t total_size  = input_size * output_size;

#if DEBUG
  if (data.weights.size()     != total_size || data.weight_grad.size() != total_size || data.biases.size() != out_dim || data.bias_grad.size() != out_dim){
    CPPDL_FATAL("linear_relu_fused::update() :: size mismatch :: ASSERTION FAILED");
  }
#endif

  const float learning_rate = eta; 
  for(size_t i = 0; i < total_size; ++i){
    data.weights[i] -= learning_rate * data.weight_grad[i]; 
  }
  for(size_t i = 0; i < output_size; ++i){
    data.biases[i] -= learning_rate * data.bias_grad[i];
  }
}

#if USE_AVX256
void neural::relu::forward(const std::vector<float> &layer_input_activations, neural::layer_data &data){
#if DEBUG
  if(layer_input_activations.size() != data.layer_input_size){
    CPPDL_FATAL("layer activations not equal to data layer input size :: ASSERTION FAILED"); 
  }
  assert(layer_input_activations.size() == data.layer_input_size); 
#endif 
  data.input = layer_input_activations;
  const size_t output_size = data.layer_output_size;
  data.output.resize(output_size); 
  float       *output_rows           = data.output.data();
  const float *activation_rows       = layer_input_activations.data();
  __m256 zero_vec                    = _mm256_set1_ps(0); 
  size_t i = 0;   
 
  for(; i + 7 < output_size; i += 8){
    __m256 activations_vec = _mm256_loadu_ps(&activation_rows[i]); 
    __m256 max_vec         = _mm256_max_ps(zero_vec, activations_vec);
    _mm256_storeu_ps(&output_rows[i], max_vec); 
  }
  for(; i < output_size; ++i){
    output_rows[i] = std::max(0.0f, activation_rows[i]); 
  }
}
#else
void neural::relu::forward(const std::vector<float> &layer_input_activations, neural::layer_data &data){
#if DEBUG
  if(layer_input_activations.size() != data.layer_input_size){
    CPPDL_FATAL("layer activations not equal to data layer input size :: ASSERTION FAILED"); 
  }
  assert(layer_input_activations.size() == data.layer_input_size); 
#endif 
  data.input = layer_input_activations;
  for(size_t i = 0; i < data.layer_output_size; ++i){
    data.output[i] = std::max(0.0f, layer_input_activations[i]);
  }
}
#endif
#if USE_AVX256 
void neural::relu::forward_batched(const std::vector<float> &input_batch, size_t batch_size, layer_data &data, std::vector<float> &output_batch){
  const size_t input_size = data.layer_input_size;

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATAL("RELU::FORWARD_BATCHED :: INPUT SIZE MISMATCH :: ASSERTION FAILED");
  }
  assert(input_batch.size() == input_size * batch_size); 
#endif 
  
  output_batch.resize(input_size * batch_size);
  float *output_batch_rows = output_batch.data();
  __m256 zero_vec         = _mm256_setzero_ps(); 
 
  size_t i = 0; 
  for(; i + 7 < input_size * batch_size; i += 8){
    __m256 input_vec = _mm256_loadu_ps(&input_batch[i]); 
    __m256 max_vec   = _mm256_max_ps(zero_vec, input_vec); 
    _mm256_storeu_ps(&output_batch_rows[i], max_vec); 
  }
  for(; i < input_size * batch_size; ++i){
    output_batch_rows[i] = (input_batch[i] > 0.0f) ? input_batch[i] : 0.0f; 
  }
  data.input  = input_batch; 
  data.output = output_batch; 
}   
#else 
void neural::relu::forward_batched(const std::vector<float> &input_batch, size_t batch_size, layer_data &data, std::vector<float> &output_batch){
  const size_t input_size = data.layer_input_size;

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATAL("RELU::FORWARD_BATCHED :: INPUT SIZE MISMATCH :: ASSERTION FAILED");
  }
  assert(input_batch.size() == input_size * batch_size); 
#endif 

  output_batch.resize(input_size * batch_size);
 
  for(size_t i = 0; i < input_size * batch_size; ++i){
    output_batch[i] = (input_batch[i] > 0.0f) ? input_batch[i] : 0.0f; 
  }
  data.input  = input_batch; 
  data.output = output_batch; 
}
#endif 


#if USE_AVX256
void neural::relu::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch){
  const size_t feature_size = data.layer_output_size;
#if DEBUG
  if(grad_output_batch.size() != feature_size * batch_size || data.output.size() != feature_size * batch_size){
    CPPDL_FATAL("RELU::BACKWARDS_BATCHED() :: size mismatch :: ASSERTION FAILED"); 
    assert(grad_output_batch.size() == feature_size * batch_size || data.output.size() == feature_size * batch_size);
  }
#endif 
  if(grad_input_batch.size() != feature_size * batch_size){
    grad_input_batch.resize(feature_size * batch_size);
  }
  data.layer_deriv_in.resize(feature_size * batch_size); 
  size_t i = 0;  
  float *grad_input_data  = grad_input_batch.data();
  float *deriv_input_data = data.layer_deriv_in.data();
  const __m256 zero_vec   = _mm256_setzero_ps();

  for(; i + 7 < feature_size * batch_size; i += 8){
    __m256 curr_data_vec   = _mm256_loadu_ps(&data.output[i]);
    __m256 d_data_vec      = _mm256_loadu_ps(&grad_output_batch[i]);
    __m256 mask_vec        = _mm256_cmp_ps(curr_data_vec, zero_vec, _CMP_GT_OQ);
    __m256 deriv_vec       = _mm256_and_ps(d_data_vec, mask_vec); 
    _mm256_storeu_ps(&grad_input_data[i], deriv_vec); 
    _mm256_storeu_ps(&deriv_input_data[i], deriv_vec); 
  }
  for(; i < feature_size * batch_size; ++i){
    float curr_data        = data.output[i];
    float d_data           = grad_output_batch[i]; 
    float deriv_out        = (curr_data > 0.0f) ? d_data : 0.0f; 
    grad_input_data[i]     = deriv_out; 
    deriv_input_data[i]    = deriv_out; 
  }
}
#else
void neural::relu::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch){
  const size_t feature_size = data.layer_output_size;
#if DEBUG
  if(grad_output_batch.size() != feature_size * batch_size || data.output.size() != feature_size * batch_size){
    CPPDL_FATAL("RELU::BACKWARDS_BATCHED() :: size mismatch :: ASSERTION FAILED"); 
    assert(grad_output_batch.size() == feature_size * batch_size || data.output.size() == feature_size * batch_size);
  }
#endif 
  if(grad_input_batch.size() != feature_size * batch_size){
    grad_input_batch.resize(feature_size * batch_size);
  }
  data.layer_deriv_in.resize(feature_size * batch_size); 

  for(size_t i = 0; i < feature_size * batch_size; ++i){
    float curr_data        = data.output[i];
    float d_data           = grad_output_batch[i]; 
    float deriv_out        = (curr_data > 0.0f) ? d_data : 0.0f; 
    grad_input_batch[i]    = deriv_out; 
    data.layer_deriv_in[i] = deriv_out; 
  }
}
#endif 

#if USE_AVX256
void neural::relu::backwards(const std::vector<float> &layer_deriv_out, layer_data &data){
#if DEBUG 
  if(layer_deriv_out.size() != data.layer_output_size){
    CPPDL_FATAL("layer derivative output size is not equal to data layer ouput size :: ASSERTION FAILED");
  }
  assert(layer_deriv_out.size() == data.layer_output_size); 
#endif
  const size_t output_size = data.layer_output_size; 
  const float *deriv_rows  = layer_deriv_out.data();
  float       *data_rows   = data.layer_deriv_in.data();
  float       *input_rows  = data.input.data(); 
  const __m256 zero_vec    = _mm256_setzero_ps(); 
  const __m256 one_vec     = _mm256_set1_ps(1.0f);
  size_t i = 0;
 
  for(; i + 7 < output_size; i += 8){
    __m256 input_vec  = _mm256_loadu_ps(&input_rows[i]);
    __m256 mask_vec   = _mm256_cmp_ps(input_vec, zero_vec, _CMP_GT_OS);
    __m256 grad_vec   = _mm256_and_ps(one_vec, mask_vec);
    __m256 deriv_vec  = _mm256_loadu_ps(&deriv_rows[i]); 
    __m256 mulled_vec = _mm256_mul_ps(grad_vec, deriv_vec); 
    _mm256_storeu_ps(&data_rows[i], mulled_vec); 
  }
  for(; i < output_size; ++i){
    data_rows[i] = deriv_rows[i] * ((input_rows[i] > 0.0f) ? 1.0f : 0.0f);
  }
  data.layer_deriv_out = layer_deriv_out;
}
#else
void neural::relu::backwards(const std::vector<float> &layer_deriv_out, layer_data &data){
#if DEBUG 
  if(layer_deriv_out.size() != data.layer_output_size){
    CPPDL_FATAL("layer derivative output size is not equal to data layer ouput size :: ASSERTION FAILED");
  }
  assert(layer_deriv_out.size() == data.layer_output_size); 
#endif
  std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f);
  for(size_t i = 0; i < data.layer_output_size; ++i){
    float grad = (data.input[i] > 0.0f) ? 1.0f : 0.0f; 
    data.layer_deriv_in[i] = layer_deriv_out[i] * grad; 
  }
  data.layer_deriv_out = layer_deriv_out; 
}
#endif

void neural::sigmoid::forward(const std::vector<float> &layer_input_activations, layer_data &data){
#if DEBUG
  if(layer_input_activations.size() != data.layer_input_size){
    CPPDL_FATAL("LAYER INPUT ACTIVATIONS NOT EQUAL TO DATA LAYER INPUT SIZE : ASSERTION FAILED");
  }
  assert(layer_input_activations.size() == data.layer_input_size); 
#endif
  data.output = functions::sigmoid(layer_input_activations); 
}

void neural::sigmoid::forward_batched(const std::vector<float> &input_batch, size_t batch_size, layer_data &data, std::vector<float> &output_batch){
 const size_t input_size = data.layer_input_size; 

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATA("SIGMOIG::FORWARD_BATCHED :: INPUT SIZE MISMATCH :: ASSERTION FAILED"); 
  }
  assert(input_batch.size() == input_size * batch_size); 
#endif 
  data.input = input_batch;
  if(output_batch.size() != input_size * batch_size){
    output_batch.resize(input_size * batch_size);
  }
  data.output.resize(input_size * batch_size); 

  for(size_t i = 0; i < input_size * batch_size; ++i){
    float input_data  = input_batch[i]; 
    float output_data = 1.0f / (1.0f + std::expf(-input_data)); 
    output_batch[i]   = output_data;
    data.output[i]    = output_data; 
  }
}

#if USE_AVX256
void neural::sigmoid::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch){
  size_t output_size = data.layer_output_size;
#if DEBUG
  if(grad_output_batch.size() != output_size * batch_size || data.output.size() != output_size * batch_size){
    CPPDL_FATAL("NEURAL::SIGMOID::BACKWARDS_BATCHED() :: size mismatch :: ASSERTION FAILED"); 
    assert(grad_output_batch.size() == output_size * batch_size || data.output.size() == output_size * batch_size); 
  }
#endif 

  if(grad_input_batch.size() != output_size * batch_size){
    grad_input_batch.resize(output_size * batch_size); 
  }
  data.layer_deriv_in.resize(output_size * batch_size);

  float *grad_input_data  = grad_input_batch.data(); 
  float *deriv_input_data = data.layer_deriv_in.data(); 
  const __m256 one_vec    = _mm256_set1_ps(1.0f); 
 
  size_t i = 0; 
  for(; i + 7 < output_size * batch_size; i += 8){
    __m256 output_data_vec   = _mm256_loadu_ps(&data.output[i]);
    __m256 d_output_data_vec = _mm256_loadu_ps(&grad_output_batch[i]); 
    __m256 sub_vec           = _mm256_sub_ps(one_vec, output_data_vec);
    __m256 left_mul_vec      = _mm256_mul_ps(d_output_data_vec, output_data_vec); 
    __m256 full_mul_vec      = _mm256_mul_ps(left_mul_vec, sub_vec);
    _mm256_storeu_ps(&grad_input_data[i], full_mul_vec); 
    _mm256_storeu_ps(&deriv_input_data[i], full_mul_vec);
  }
  for(; i < output_size * batch_size; ++i){
    float output_data      = data.output[i]; 
    float d_output_data    = grad_output_batch[i];
    float d_input_data     = d_output_data * output_data * (1.0f - output_data); 
    grad_input_data[i]     = d_input_data; 
    deriv_input_data[i]    = d_input_data; 
  }
}
#else
void neural::sigmoid::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch){
  size_t output_size = data.layer_output_size;
#if DEBUG
  if(grad_output_batch.size() != output_size * batch_size || data.output.size() != output_size * batch_size){
    CPPDL_FATAL("NEURAL::SIGMOID::BACKWARDS_BATCHED() :: size mismatch :: ASSERTION FAILED"); 
    assert(grad_output_batch.size() == output_size * batch_size || data.output.size() == output_size * batch_size); 
  }
#endif 

  if(grad_input_batch.size() != output_size * batch_size){
    grad_input_batch.resize(output_size * batch_size); 
  }
  data.layer_deriv_in.resize(output_size * batch_size);

  for(size_t i = 0; i < output_size * batch_size; ++i){
    float output_data      = data.output[i]; 
    float d_output_data    = grad_output_batch[i];
    float d_input_data     = d_output_data * output_data * (1.0f - output_data); 
    grad_input_batch[i]    = d_input_data; 
    data.layer_deriv_in[i] = d_input_data; 
  }
}
#endif 

#if USE_AVX256
void neural::sigmoid::backwards(const std::vector<float> &layer_deriv_out, neural::layer_data &data){
#if DEBUG 
  if(layer_deriv_out.size() != data.layer_output_size){
    CPPDL_FATAL("layer derivative output size is not equal to data layer ouput size :: ASSERTION FAILED");
  }
  assert(layer_deriv_out.size() == data.layer_output_size); 
#endif

  const size_t output_size    = data.layer_output_size; 
  float        *input_vec     = data.layer_deriv_in.data();
  const float  *data_vec      = data.output.data(); 
  const float  *deriv_out_vec = layer_deriv_out.data();
  const __m256 one_vec        = _mm256_set1_ps(1.0f); 
  size_t i = 0; 
 
  for(; i + 7 < output_size; i += 8){
    __m256 v_vec      = _mm256_loadu_ps(&data_vec[i]); 
    __m256 sub_vec    = _mm256_sub_ps(one_vec, v_vec);
    __m256 dv_vec     = _mm256_mul_ps(v_vec, sub_vec); 
    __m256 deriv_vec  = _mm256_loadu_ps(&deriv_out_vec[i]);
    __m256 mulled_vec = _mm256_mul_ps(deriv_vec, dv_vec); 
    _mm256_storeu_ps(&input_vec[i], mulled_vec); 
  }
  for(;i < output_size; ++i){
    input_vec[i] = ((data_vec[i] * (1.0f - data_vec[i])) * deriv_out_vec[i]); 
  }
  data.layer_deriv_out = layer_deriv_out; 
}
#else
void neural::sigmoid::backwards(const std::vector<float> &layer_deriv_out, neural::layer_data &data){
#if DEBUG 
  if(layer_deriv_out.size() != data.layer_output_size){
    CPPDL_FATAL("layer derivative output size is not equal to data layer ouput size :: ASSERTION FAILED");
  }
  assert(layer_deriv_out.size() == data.layer_output_size); 
#endif
  std::fill(data.layer_deriv_in.begin(), data.layer_deriv_in.end(), 0.0f); 
  for(size_t i = 0; i < data.layer_output_size; ++i){
    float v = data.output[i]; 
    float dv = v * (1.0f - v); 
    data.layer_deriv_in[i] = layer_deriv_out[i] * dv; 
  }
  data.layer_deriv_out = layer_deriv_out; 
}
#endif

void neural::linear_sigmoid_fused::forward_batched(const std::vector<float> &input_batch, size_t batch_size, layer_data &data, std::vector<float> &output_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size;

#if DEBUG
  if(input_batch.size() != input_size * batch_size){
    CPPDL_FATAL("linear_sigmoid_fused::forward_batched() :: input size mismatch :: ASSERTION FAILED"); 
  }
  assert(input_batch.size() == input_size * batch_size); 
#endif 

  data.input = input_batch; 
  if(output_batch.size() != output_size * batch_size){
    output_batch.assign(output_size * batch_size, 0.0f);
  }else{
    std::fill(output_batch.begin(), output_batch.end(), 0.0f); 
  }

  level3::mat_ops_view mat_w { output_size, input_size, input_size, data.weights.data()                    };
  level3::mat_ops_view mat_x { input_size , batch_size, batch_size, const_cast<float*>(input_batch.data()) };
  level3::mat_ops_view mat_y { output_size, batch_size, batch_size, output_batch.data()                    };
  
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, mat_w, mat_x, 1.0f, 0.0f, mat_y); 
  
  for(size_t i = 0; i < output_size; ++i){
    float *data_row    = &output_batch[i * batch_size]; 
    const float biases = data.biases[i]; 
    for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
      float output_vals     = data_row[batch_index] + biases; 
      float sigmoid_val     = 1.0f / (1.0f + std::expf(-output_vals)); 
      data_row[batch_index] = sigmoid_val; 
    }
  }
  data.output = output_batch;
}

void neural::linear_sigmoid_fused::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, layer_data &data, std::vector<float> &grad_input_batch){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size; 
  const size_t total_size  = output_size * batch_size; 

#if DEBUG 
  if(grad_output_batch.size() != total_size || data.output.size() != total_size || data.input.size() != input_size * batch_size){
    CPPDL_FATAL("neural::linear_sigmoid_fused::backwards_batched() :: size mismatch :: ASSERTION FAILED"); 
  }
  assert(grad_output_batch.size() == total_size && data.output.size() == total_size && data.input.size() == input_size * batch_size)
#endif

  std::vector<float> function_grad(total_size);
  float *function_grad_data = function_grad.data();
  __m256 one_vec            = _mm256_set1_ps(1.0f);  
  
  size_t i = 0; 
  for(; i + 7 < total_size; i += 8){
    __m256 output_vec   = _mm256_loadu_ps(&data.output[i]); 
    __m256 d_output_vec = _mm256_loadu_ps(&grad_output_batch[i]); 
    __m256 mul_vec      = _mm256_mul_ps(d_output_vec, output_vec); 
    __m256 sub_vec      = _mm256_sub_ps(one_vec, output_vec); 
    __m256 final_vec    = _mm256_mul_ps(mul_vec, sub_vec);
    _mm256_storeu_ps(&function_grad_data[i], final_vec); 
  }
  for(;i < total_size; ++i){
    float final_val = grad_output_batch[i] * data.output[i] * (1.0f - data.output[i]);
    function_grad_data[i] = final_val; 
  }

  data.layer_deriv_out = function_grad;

  if(data.weight_grad.size() != output_size * input_size){
    data.weight_grad.resize(output_size * input_size, 0.0f); 
  }else{
    std::fill(data.weight_grad.begin(), data.weight_grad.end(), 0.0f); 
  }
  
  level3::mat_ops_view function_grad_view { output_size, batch_size, batch_size, function_grad.data()    };
  level3::mat_ops_view mat_x_view         { input_size , batch_size, batch_size, data.input.data()       };
  level3::mat_ops_view dw_view            { output_size, input_size, input_size, data.weight_grad.data() };

  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::transpose, function_grad_view, mat_x_view, 1.0f, 0.0f, dw_view);

  if(grad_input_batch.size() != batch_size * input_size){
    grad_input_batch.resize(batch_size * input_size, 0.0f); 
  }else{
    std::fill(grad_input_batch.begin(), grad_input_batch.end(), 0.0f); 
  }

  level3::mat_ops_view mat_w_view { output_size, input_size, input_size, data.weights.data()             };
  level3::mat_ops_view dx_view    { input_size , batch_size, batch_size, grad_input_batch.data()         };

  level3::blas::crush_gemm(level3::transpose_gemm::transpose, level3::transpose_gemm::no_transpose, mat_w_view, function_grad_view, 1.0f, 0.0f, dx_view);

  data.layer_deriv_in = grad_input_batch; 

  if(data.bias_grad.size() != output_size){
    data.bias_grad.assign(output_size, 0.0f); 
  }else{
    std::fill(data.bias_grad.begin(), data.bias_grad.end(), 0.0f); 
  }

  for(size_t i = 0; i < output_size; ++i){
    const float *row_grad_data = &function_grad[i * batch_size]; 
    float curr_sum = 0.0f; 

    for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
      curr_sum += row_grad_data[batch_index]; 
    }
    data.bias_grad[i] = curr_sum;
  }
}

void neural::linear_sigmoid_fused::update(layer_data &data, float eta){
  const size_t input_size  = data.layer_input_size; 
  const size_t output_size = data.layer_output_size; 
  const size_t total_size  = input_size * output_size;

#if DEBUG
  if (data.weights.size()     != total_size || data.weight_grad.size() != total_size || data.biases.size() != out_dim || data.bias_grad.size() != out_dim){
    CPPDL_FATAL("linear_sigmoid_fused::update() :: size mismatch :: ASSERTION FAILED");
  }
#endif

  const float learning_rate = eta; 
  for(size_t i = 0; i < total_size; ++i){
    data.weights[i] -= learning_rate * data.weight_grad[i]; 
  }
  for(size_t i = 0; i < output_size; ++i){
    data.biases[i] -= learning_rate * data.bias_grad[i];
  }
}


#if USE_AVX256
float neural::mse_loss::forward_batched(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, size_t num_outputs, size_t batch_size){
  const size_t total_size = num_outputs * batch_size;

#if DEBUG 
  if(layer_preds.size() != total_size || layer_target_vals.size() != total_size){
    CPPDL_FATAL("NEURAL::MSE_LOSS::FORWARD_BATCHED() :: size mismatch :: ASSERTION FAILED"); 
  }
#endif

  const float *layer_preds_data = layer_preds.data(); 
  const float *target_vals_data = layer_target_vals.data();

  size_t i = 0;
  __m256 sum_vec = _mm256_setzero_ps(); 
  for(; i + 7 < total_size; i += 8){
    __m256 preds_vec   = _mm256_loadu_ps(&layer_preds[i]); 
    __m256 targets_vec = _mm256_loadu_ps(&layer_target_vals[i]);
    __m256 sub_vec     = _mm256_sub_ps(preds_vec, targets_vec);
    sum_vec            = _mm256_fmadd_ps(sub_vec, sub_vec, sum_vec); 
  }
  alignas(32) float temp_buff[8]; 
  _mm256_store_ps(temp_buff, sum_vec);
  double squared_sum = 0.0f;
  
  for(size_t k = 0; k < 8; ++k){
    squared_sum += temp_buff[k]; 
  }

  for(; i < total_size; ++i){
    float err_diff =  layer_preds_data[i] - target_vals_data[i]; 
    squared_sum    += double(err_diff) * double(err_diff); 
  }

  double denominator = double(total_size); 
  return static_cast<float>(squared_sum / denominator); 
} 
#endif 

#if USE_AVX256
float neural::mse_loss::forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals){
#if DEBUG
  if(layer_preds.size() != layer_target_vals.size()){
    CPPDL_FATAL("layer predictions size not equal to layer target values size :: ASSERTION FAILED"); 
  }
  assert(layer_preds.size() == layer_target_vals.size());
#endif  

  const size_t preds_size  = layer_preds.size(); 
  const float *preds_vec  = layer_preds.data(); 
  const float *target_vec = layer_target_vals.data();
  const __m256 div_vec    = _mm256_set1_ps(0.5f);
  __m256 loss_vec         = _mm256_setzero_ps(); 
  size_t i = 0; 
  float loss = 0; 
 
  for(; i + 7 < preds_size; i += 8){
    __m256 preds_in_vec     = _mm256_loadu_ps(&preds_vec[i]); 
    __m256 layer_target_vec = _mm256_loadu_ps(&target_vec[i]);
    __m256 diff_vec         = _mm256_sub_ps(preds_in_vec, layer_target_vec);
    __m256 squared_vec      = _mm256_mul_ps(diff_vec, diff_vec); 
    loss_vec                = _mm256_fmadd_ps(squared_vec, div_vec, loss_vec);
  }
  alignas(32) float temp[8]; 
  _mm256_store_ps(temp, loss_vec);
  loss = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  for(; i < preds_size; ++i){
    loss += 0.5f * ( (preds_vec[i] - target_vec[i]) * (preds_vec[i] - target_vec[i]) );     
  }
  return loss;
}
#else
float neural::mse_loss::forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals){
#if DEBUG
  if(layer_preds.size() != layer_target_vals.size()){
    CPPDL_FATAL("layer predictions size not equal to layer target values size :: ASSERTION FAILED"); 
  }
  assert(layer_preds.size() == layer_target_vals.size());
#endif  
  float loss = 0.0f; 
  for(size_t i = 0; i < layer_preds.size(); ++i){
    float diff = layer_preds[i] - layer_target_vals[i]; 
    loss += 0.5f * diff * diff; 
  }
  return loss; 
}
#endif

#if USE_AVX256
void neural::mse_loss::backwards_batched(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, size_t num_outputs, size_t batch_size, std::vector<float> &grad_preds){
  const size_t total_size = num_outputs * batch_size; 
#if DEBUG
  if(layer_preds.size() != total_size || layer_target_vals.size() != total_size){
    CPPDL_FATAL("NEURAL::MSE_LOSS::BACKWARDS_BATCHED() :: size mismatch :: ASSERTION FAILED"); 
    assert(layer_preds.size() == total_size || layer_target_vals.size() == total_size);   
  }
#endif 
  grad_preds.resize(total_size); 
  const float *layer_preds_data = layer_preds.data(); 
  const float *target_vals_data  = layer_target_vals.data(); 
  float       *grad_preds_data  = grad_preds.data();

  const float grad_scale        = 2.0f / static_cast<float>(total_size);
  __m256      grad_scale_vec    = _mm256_set1_ps(grad_scale);

  size_t i = 0; 
  for(; i + 7 < total_size; i += 8){
    __m256 layer_preds_vec = _mm256_loadu_ps(&layer_preds[i]); 
    __m256 target_vals_vec = _mm256_loadu_ps(&layer_target_vals[i]); 
    __m256 sub_vec         = _mm256_sub_ps(layer_preds_vec, target_vals_vec); 
    __m256 mul_vec         = _mm256_mul_ps(sub_vec, grad_scale_vec);
    _mm256_storeu_ps(&grad_preds_data[i], mul_vec); 
  }
  for(; i < total_size; ++i){
    grad_preds_data[i] = grad_scale * (layer_preds_data[i] - target_vals_data[i]); 
  }
}
#endif 

#if USE_AVX256
void neural::mse_loss::backwards(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out){
#if DEBUG
  if(layer_preds.size() != layer_target_vals.size()){
    CPPDL_FATAL("layer predictions size not equal to layer target values size :: ASSERTION FAILED"); 
  }
  assert(layer_preds.size() == layer_target_vals.size());
#endif 

  layer_deriv_out.resize(layer_preds.size());
  const size_t preds_size   = layer_preds.size();
  const float *preds_data  = layer_preds.data(); 
  const float *target_data = layer_target_vals.data(); 
  float       *deriv_data  = layer_deriv_out.data();
  size_t i = 0; 

  for(; i + 7 < preds_size; i += 8){
    __m256 preds_vec  = _mm256_loadu_ps(&preds_data[i]); 
    __m256 target_vec = _mm256_loadu_ps(&target_data[i]); 
    __m256 sub_vec    = _mm256_sub_ps(preds_vec, target_vec); 
    _mm256_storeu_ps(&deriv_data[i], sub_vec); 
  }
  for(; i < preds_size; ++i){
    deriv_data[i] = preds_data[i] - target_data[i]; 
  }
}
#else
void neural::mse_loss::backwards(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out){
#if DEBUG
  if(layer_preds.size() != layer_target_vals.size()){
    CPPDL_FATAL("layer predictions size not equal to layer target values size :: ASSERTION FAILED"); 
  }
  assert(layer_preds.size() == layer_target_vals.size());
#endif 
  layer_deriv_out.resize(layer_preds.size()); 
  for(size_t i = 0; i < layer_preds.size(); ++i){
    layer_deriv_out[i] = layer_preds[i] - layer_target_vals[i]; 
  } 
}
#endif

float neural::cross_entropy_loss_logits::forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals){
#if DEBUG
  if(layer_preds.size() != layer_target_vals.size()){
    CPPDL_FATAL("layer predictions size not equal to layer target values size :: ASSERTION FAILED"); 
  }
  assert(layer_preds.size() == layer_target_vals.size());
#endif  
  float loss = 0.0f;
  float epsi = 1e-15; 
  for(size_t i = 0; i < layer_preds.size(); ++i){
    float temp = std::max(layer_preds[i], epsi); 
    loss -= layer_target_vals[i] * std::log(temp); 
  }
  return loss; 
}

float neural::cross_entropy_loss_logits::forward_batched(const std::vector<float> &logits_data, const std::vector<float> &layer_target_vals, size_t num_classes, size_t batch_size){
  const size_t total_size = num_classes * batch_size; 
#if DEBUG
  if(logits_data.size() != total_size){
    CPPDL_FATAL("neural::cross_entropy_loss_logits::forward_batched() :: logits size mismatch :: ASSERTION FAILED"); 
  } 
  if(layer_target_vals.size() != batch_size){
    CPPDL_FATAL("neural::cross_entropy_loss_logits::forward_batched() :: targets size mismatch :: ASSERTION FAILED");
  }
  assert(logits_data.size() == total_size);
  assert(layer_target_vals.size() == batch_size); 
#endif 
  const float *logits          = logits_data.data();
  double       logits_loss_sum = 0.0f;
  for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
    float max_logits_val = -std::numeric_limits<float>::infinity(); 
    for(size_t class_index = 0; class_index < num_classes; ++class_index){
      float logits_class_batch = logits[class_index * batch_size + batch_index]; 
      if(logits_class_batch > max_logits_val){
        max_logits_val = logits_class_batch; 
      }
    } 

    double exponential_sum = 0.0f; 
    for(size_t class_index = 0; class_index < num_classes; ++class_index){
      float logits_class_batch = logits[class_index * batch_size + batch_index]; 
      exponential_sum += std::exp(double(logits_class_batch - max_logits_val)); 
    }

    int output_batch = layer_target_vals[batch_index];
#if DEBUG
    if(output_batch < 0 || static_cast<size_t>(output_batch) >= num_classes){
      CPPDL_FATAL("neural::cross_entropy_loss_logits::forward_batched() :: target out of range :: ASSERTION FAILED"); 
    }
    assert(output_batch > 0 || static_cast<size_t>(output_batch) < num_classes); 
#endif
    float logits_output_batch = logits[output_batch * batch_size + batch_index];
    double log_probability    = double(logits_output_batch - max_logits_val) - std::log(exponential_sum); 
    logits_loss_sum += -log_probability; 
  }
  
  double denominator = double(batch_size); 
  return static_cast<float>(logits_loss_sum / denominator);
}


void neural::cross_entropy_loss_logits::backwards_batched(const std::vector<float> &logits_data, const std::vector<float> &layer_target_vals, size_t num_classes, size_t batch_size, std::vector<float> &grad_logits){
  const size_t total_size = num_classes * batch_size; 
#if DEBUG
  if(logits_data.size() != total_size){
    CPPDL_FATAL("neural::cross_entropy_loss_logits::forward_batched() :: logits size mismatch :: ASSERTION FAILED"); 
  } 
  if(layer_target_vals.size() != batch_size){
    CPPDL_FATAL("neural::cross_entropy_loss_logits::forward_batched() :: targets size mismatch :: ASSERTION FAILED");
  }
  assert(logits_data.size() == total_size);
  assert(layer_target_vals.size() == batch_size); 
#endif 
  grad_logits.assign(total_size, 0.0f);
  const float *logits       = logits_data.data(); 
  float       *grad_data    = grad_logits.data();

  const float inverse_batch = 1.0f / static_cast<float>(batch_size); 

  std::vector<float> logits_prob(num_classes); 
  for(size_t batch_index = 0; batch_index < batch_size; ++batch_index){
    float max_logits_val = -std::numeric_limits<float>::infinity(); 
    for(size_t class_index = 0; class_index < num_classes; ++class_index){
      float logits_class_batch = logits[class_index * batch_size + batch_index]; 
      if(logits_class_batch > max_logits_val){
        max_logits_val = logits_class_batch; 
      }
    }

    double exponential_sum = 0.0f; 
    for(size_t class_index = 0; class_index < num_classes; ++class_index){
      float logits_class_batch  = logits[class_index * batch_size + batch_index]; 
      float eno_exp             = std::exp(logits_class_batch - max_logits_val); 
      logits_prob[class_index]  = eno_exp; 
      exponential_sum          += eno_exp; 
    }

    float inverse_sum = 1.0f / static_cast<float>(exponential_sum);
    for(size_t class_index = 0; class_index < num_classes; ++class_index){
      logits_prob[class_index] *= inverse_sum; 
    }
    int output_batch = layer_target_vals[batch_index]; 

#if DEBUG
    if(output_batch < 0 || static_cast<size_t>(output_batch) >= num_classes){
      CPPDL_FATAL("neural::cross_entropy_loss_logits::forward_batched() :: target out of range :: ASSERTION FAILED"); 
    }
    assert(output_batch > 0 || static_cast<size_t>(output_batch) < num_classes); 
#endif
    for(size_t class_index = 0; class_index < num_classes; ++class_index){
      float one_hot = (class_index == static_cast<size_t>(output_batch)) ? 1.0f : 0.0f;
      grad_data[class_index * batch_size + batch_index] = (logits_prob[class_index] - one_hot) * inverse_batch; 
    }   
  }
}

void neural::cross_entropy_loss_logits::backwards(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out){
#if DEBUG
  if(layer_preds.size() != layer_target_vals.size()){
    CPPDL_FATAL("layer predictions size not equal to layer target values size :: ASSERTION FAILED"); 
  }
  assert(layer_preds.size() == layer_target_vals.size());
#endif 
layer_deriv_out.resize(layer_preds.size()); 
  for(size_t i = 0; i < layer_preds.size(); ++i){
    layer_deriv_out[i] = layer_preds[i] - layer_target_vals[i]; 
  }
}

void neural::nn::add_linear(size_t input, size_t output){
  data_layer.emplace_back(input, output); 
  layers.push_back(std::make_unique<linear>()); 
}

void neural::nn::add_linear_relu_fused(size_t input_size, size_t output_size){
  data_layer.emplace_back(input_size, output_size); 
  layers.push_back(std::make_unique<linear_relu_fused>()); 
}

void neural::nn::add_linear_sigmoid_fused(size_t input_size, size_t output_size){
  data_layer.emplace_back(input_size, output_size); 
  layers.push_back(std::make_unique<linear_sigmoid_fused>()); 
}

void neural::nn::add_relu(size_t input){
  data_layer.emplace_back(input, input);
  layers.push_back(std::make_unique<relu>()); 
}

void neural::nn::add_sigmoid(size_t input){
  data_layer.emplace_back(input, input);
  layers.push_back(std::make_unique<sigmoid>()); 
}

void neural::nn::add_loss(std::unique_ptr<neural::loss> loss_in){
  layer_loss_function = std::move(loss_in);
}

void neural::nn::zero_grad(neural::nn &net_in){
  for(size_t index = 0; index < net_in.layers.size(); ++index){
    for(size_t i = 0; i < net_in.data_layer[index].weight_grad.size(); ++i){
      net_in.data_layer[index].weight_grad[i] = 0; 
    }
    for(size_t i = 0; i < net_in.data_layer[index].bias_grad.size(); ++i){
      net_in.data_layer[index].bias_grad[i] = 0; 
    }
  }
}

std::vector<float> neural::nn::forward(const std::vector<float> &layer_input_vals){
#if DEBUG
  if(layers.empty()){
    CPPDL_FATAL("LAYER VECTOR IS EMPTY :: ASSERTION FAILED");
  }
  assert(!layers.empty()); 
#endif 
  std::vector<float> current = layer_input_vals; 
  for(size_t i = 0; i < layers.size(); ++i){
    layers[i]->forward(current, data_layer[i]); 
    current = data_layer[i].output; 
  }
  return current; 
}

std::vector<float> neural::nn::forward_batched(const std::vector<float> &input_batch, size_t batch_size, std::vector<float> &output_batch){
#if DEBUG
  if(layers.empty()){
    CPPDL_FATAL("LAYER VECTOR IS EMPTY :: ASSERTION FAILED"); 
  }
  assert(!layers.empty());
#endif
  std::vector<float> current_batch = input_batch;
  std::vector<float> next_batch    = output_batch; 
  for(size_t i = 0; i < layers.size(); ++i){
    layers[i]->forward_batched(current_batch, batch_size,data_layer[i], next_batch); 
    current_batch.swap(next_batch); 
  }
  output_batch = current_batch; 
  return output_batch; 
}

void neural::nn::backwards(const std::vector<float> &layer_target_vals){
  std::vector<float> grad = layer_target_vals; 
  for(int i = static_cast<int>(layers.size() - 1); i >=0; --i){
    layers[i]->backwards(grad, data_layer[i]); 
    grad = data_layer[i].layer_deriv_in; 
  }
}

void neural::nn::backwards_batched(const std::vector<float> &grad_output_batch, size_t batch_size, std::vector<float> &grad_input_batch){
  std::vector<float> curr_grad_vec = grad_output_batch;
  std::vector<float> next_grad_vec;

  for(int i = static_cast<int>(layers.size() - 1); i >= 0; --i){
    next_grad_vec.clear();
    layers[i]->backwards_batched(curr_grad_vec, batch_size, data_layer[i], next_grad_vec);
    data_layer[i].layer_deriv_in = next_grad_vec; 
    curr_grad_vec.swap(next_grad_vec);
  }

  grad_input_batch = std::move(curr_grad_vec); 
}

float neural::nn::get_loss(const std::vector<float> &layer_target_vals){
#if DEBUG
  if(!layer_loss_function){
    CPPDL_FATAL("no loss function attached :: ASSERTION FAILED");
    assert(layer_loss_function); 
    return 0.0f; 
  }
#endif 
  return layer_loss_function->forward(data_layer.back().output, layer_target_vals);
}


float neural::nn::get_loss_batched(const std::vector<float> &target_batch, size_t batch_size){
#if DEBUG
  if(!layer_loss_function){
    CPPDL_FATAL("no layer_loss_function attached :: ASSERTION FAILED"); 
  }
  assert(layer_loss_function); 
  return {}; 
#endif
  const neural::layer_data &last_data   = data_layer.back(); 
  const size_t              output_size = last_data.layer_output_size; 
#if DEBUG
  if(last_data.output.size() != output_size * batch_size){
    CPPDL_FATAL("neural::nn::get_loss_batched() :: output_size != output_features * batch_size :: ASSERTION FAILED"); 
  }
  if(target_batch.size() != output_size * batch_size){
    CPPDL_FATAL("neural::nn::get_loss_batched() :: target_batch size != output_features * batch_size :: ASSERTION FAILED"); 
  }
  assert(last_data.output,size() == output_size * batch_size);
  assert(target_batch.size() == output_size * batch_size); 
#endif

  return layer_loss_function->forward_batched(last_data.output, target_batch, output_size, batch_size); 
}

std::vector<float> neural::nn::get_grad(const std::vector<float> &layer_target_vals){
#if DEBUG
  if(!layer_loss_function){
    CPPDL_FATAL("no loss function attached :: ASSERTION FAILED");
    assert(layer_loss_function); 
    return {}; 
  }
#endif 
  std::vector<float> curr_layer_deriv_out;
  layer_loss_function->backwards(data_layer.back().output, layer_target_vals, curr_layer_deriv_out);
  return curr_layer_deriv_out; 
}

void neural::nn::update(float eta){
  for(size_t i = 0; i < layers.size(); ++i){
    layers[i]->update(data_layer[i], eta); 
  }
}

std::vector<float> neural::nn::get_grad_batched(const std::vector<float> &target_batch, size_t batch_size){
#if DEBUG
  if(!layer_loss_function){
    CPPDL_FATAL("no layer_loss_function attached :: ASSERTION FAILED"); 
  }
  assert(layer_loss_function); 
  return {}; 
#endif 
  const neural::layer_data &last_data   = data_layer.back(); 
  const size_t              output_size = last_data.layer_output_size;

#if DEBUG
  if(last_data.output.size() != output_size * batch_size){
    CPPDL_FATAL("neural::nn::get_grad_batched() :: output_size != output_features * batch_size :: ASSERTION FAILED"); 
  }
  if(target_batch.size() != output_size * batch_size){
    CPPDL_FATAL("neural::nn::get_grad_batched() :: target_batch size != output_features * batch_size :: ASSERTION FAILED"); 
  }
  assert(last_data.output,size() == output_size * batch_size);
  assert(target_batch.size() == output_size * batch_size); 
#endif
  std::vector<float> grad_output_batch;
  layer_loss_function->backwards_batched(last_data.output, target_batch, output_size, batch_size, grad_output_batch); 
  return grad_output_batch; 
}

void neural::nn::draw_load_bar(int x){
  float prog = (float)x / 2500; 
  float bw = 90;
  float pos = bw * prog;
  for(size_t i = 0; i < bw; ++i){
    if(i < pos) std::cout << "\033[35;106m \033[m"; 
    else std::cout << " "; 
  }
  std::cout<<"] " << prog * 100 << "%\r" << std::flush;
}

uint64_t neural::nn::nanos() {
  struct timespec start;

  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}
