#include "../../include/neural_core/neural_network.hpp"

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
    bias_grad(output,0.0f){
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
void neural::relu::backwards(const std::vector<float> &layer_deriv_out, neural::layer_data &data){
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
void neural::relu::backwards(const std::vector<float> &layer_deriv_out, neural::layer_data &data){
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

void neural::sigmoid::forward(const std::vector<float> &layer_input_activations, neural::layer_data &data){
#if DEBUG
  if(layer_input_activations.size() != data.layer_input_size){
    CPPDL_FATAL("LAYER INPUT ACTIVATIONS NOT EQUAL TO DATA LAYER INPUT SIZE : ASSERTION FAILED");
  }
  assert(layer_input_activations.size() == data.layer_input_size); 
#endif
  data.output = functions::sigmoid(layer_input_activations); 
}

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
    //layer_deriv_out[i] = layer_preds[i] - layer_target_vals[i]; 
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

float neural::cross_entropy_loss::forward(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals){
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

void neural::cross_entropy_loss::backwards(const std::vector<float> &layer_preds, const std::vector<float> &layer_target_vals, std::vector<float> &layer_deriv_out){
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
    CPPDL_FATAL("layer vector is empty :: ASSERTION FAILED");
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

void neural::nn::backwards(const std::vector<float> &layer_target_vals){
  std::vector<float> grad = layer_target_vals; 
  for(int i = static_cast<int>(layers.size() - 1); i >=0; --i){
    layers[i]->backwards(grad, data_layer[i]); 
    grad = data_layer[i].layer_deriv_in; 
  }
}

float neural::nn::get_loss(const std::vector<float> &layer_target_vals){
#if DEBUG
  if(!layer_loss_function){
    CPPDL_FATAL("no loss function attached :: ASSERTION FAILED");
    asser(layer_loss_function); 
    return 0.0f; 
  }
#endif 
  return layer_loss_function->forward(data_layer.back().output, layer_target_vals);
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


