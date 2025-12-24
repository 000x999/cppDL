#include "neural_core/neural_network.hpp"

neural::linear::linear(size_t input, size_t output){
  weight_data.input_features  = input; 
  weight_data.output_features = output; 
}

void neural::linear::init(alloc_pool &persistent_arena){
  size_t input       = weight_data.input_features; 
  size_t output      = weight_data.output_features; 
 
  float *weights_ptr = persistent_arena.arena.nn_alloc<float>(input * output);
  float *biases_ptr  = persistent_arena.arena.nn_alloc<float>(output);
  
  auto limit = std::sqrt(6 / float(input * output));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-limit, limit); 
  
  size_t total_size = input * output; 
      
  for(size_t i = 0; i < total_size; ++i){
    weights_ptr[i] = dist(gen);  
  }

  for(size_t i = 0; i < output; ++i){
    biases_ptr[i] = 0.0f;
  }

  weight_data.weights.tensor.tensor_data    = weights_ptr;
  weight_data.weights.tensor.shape.dims[0]  = input;
  weight_data.weights.tensor.shape.dims[1]  = output; 
  weight_data.weights.tensor.shape.ndim     = 2;

  weight_data.biases.tensor.tensor_data     = biases_ptr;
  weight_data.biases.tensor.shape.dims[0]   = 1;
  weight_data.biases.tensor.shape.dims[1]   = output; 
  weight_data.biases.tensor.shape.ndim      = 2;
}

neural::neural_view neural::linear::forward(neural_view &input_view, alloc_pool &arena){
  size_t batch_size      = input_view.tensor.shape.dims[0];
  size_t input_features  = input_view.tensor.shape.dims[1];
  size_t output_features = weight_data.output_features; 

  float *output_ptr      = arena.arena.nn_alloc<float>(batch_size * output_features); 
    
  level3::mat_ops_view w_view = {
    .row_view          = batch_size, 
    .col_view          = input_features, 
    .leading_dimension = input_features, 
    .data_view         = input_view.tensor.tensor_data, 
  }; 

  level3::mat_ops_view x_view = {
    .row_view          = input_features, 
    .col_view          = output_features, 
    .leading_dimension = output_features, 
    .data_view         = weight_data.weights.tensor.tensor_data, 
  };

  level3::mat_ops_view c_view = {
    .row_view          = batch_size, 
    .col_view          = output_features, 
    .leading_dimension = output_features, 
    .data_view         = output_ptr,
  };

  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, w_view, x_view, 1.0f, 0.0f, c_view);

  float *bias_ptr = weight_data.biases.tensor.tensor_data; 
  for(size_t i = 0; i < batch_size; ++i){
    float *row_ptr = output_ptr + ( i * output_features );
    size_t j = 0; 

    for(; j + 7 < output_features; j += 8){
      __m256 row_vec  = _mm256_loadu_ps ( &row_ptr[j]          );
      __m256 bias_vec = _mm256_loadu_ps ( &bias_ptr[j]         );
      __m256 add_vec  = _mm256_add_ps  ( row_vec, bias_vec    );
      _mm256_storeu_ps                  ( &row_ptr[j], add_vec ); 
    }
    
    for(; j < output_features; ++j){
      row_ptr[j] += bias_ptr[j];
    }
  }

  neural_view output_view; 
  output_view.tensor.tensor_data      = output_ptr; 
  output_view.tensor.shape.ndim       = 2; 
  output_view.tensor.shape.dims[0]    = batch_size; 
  output_view.tensor.shape.dims[1]    = output_features; 
  output_view.tensor.shape.strides[0] = output_features; 
  output_view.tensor.shape.strides[1] = 1;
  
  return output_view;
}

#if USE_AVX256
neural::neural_view neural::relu::forward(neural_view &input_view, alloc_pool &arena){
  size_t batch_size      = input_view.tensor.shape.dims[0]; 
  size_t input_features  = input_view.tensor.shape.dims[1];
  size_t output_features = input_view.tensor.shape.dims[1]; 
  float  *data_ptr       = input_view.tensor.tensor_data; 

  /*Not sure if they need their own arenas right now, seems useless though*/
  //float *output_ptr      = arena.arena.nn_alloc<float>(batch_size * input_features);
  
  __m256 zero_vec   = _mm256_setzero_ps(); 
  size_t i          = 0; 
  size_t total_size = output_features * batch_size;  
  for(; i + 7 < total_size; i += 8){
    __m256 data_vec = _mm256_loadu_ps(&data_ptr[i]); 
    __m256 max_vec  = _mm256_max_ps(zero_vec, data_vec); 
    _mm256_storeu_ps(&data_ptr[i], max_vec); 
  }
  for(; i < total_size; ++i){
    data_ptr[i] = std::max(0.0f, data_ptr[i]);   
  }
 
  //output_ptr                          = data_ptr;
  
  neural_view output_view; 
  output_view.tensor.tensor_data      = input_view.tensor.tensor_data; 
  output_view.tensor.shape.ndim       = 2; 
  output_view.tensor.shape.dims[0]    = batch_size; 
  output_view.tensor.shape.dims[1]    = output_features; 
  output_view.tensor.shape.strides[0] = output_features; 
  output_view.tensor.shape.strides[1] = 1;
  
  return output_view; 
}
#else
neural::neural_view neural::relu::forward(neural_view &input_view, alloc_pool &arena){
  size_t batch_size      = input_view.tensor.shape.dims[0]; 
  size_t input_features  = input_view.tensor.shape.dims[1];
  size_t output_features = input_view.tensor.shape.dims[1]; 
  float  *data_ptr       = input_view.tensor.tensor_data; 

  /*Not sure if they need their own arenas right now, seems useless though*/
  //float *output_ptr      = arena.arena.nn_alloc<float>(batch_size * input_features);
  size_t total_size = output_features * batch_size; 
  for(size_t i = 0; i < total_size; ++i){
    data_ptr[i] = std::max(0.0f, data_ptr[i]);   
  }
 
  //output_ptr                          = data_ptr;
  
  neural_view output_view; 
  output_view.tensor.tensor_data      = input_view.tensor.tensor_data; 
  output_view.tensor.shape.ndim       = 2; 
  output_view.tensor.shape.dims[0]    = batch_size; 
  output_view.tensor.shape.dims[1]    = output_features; 
  output_view.tensor.shape.strides[0] = output_features; 
  output_view.tensor.shape.strides[1] = 1;
  
  return output_view; 
}
#endif

neural::neural_view neural::sigmoid::forward(neural_view &input_view, alloc_pool &arena){
  size_t batch_size      = input_view.tensor.shape.dims[0]; 
  size_t input_features  = input_view.tensor.shape.dims[1];
  size_t output_features = input_view.tensor.shape.dims[1]; 
  float  *data_ptr       = input_view.tensor.tensor_data; 

  /*Not sure if they need their own arenas right now, seems useless though*/
  //float *output_ptr      = arena.arena.nn_alloc<float>(batch_size * input_features);
  size_t total_size = batch_size * output_features; 
  for(size_t i = 0; i < total_size; ++i){
    data_ptr[i] = (1 / (1 + std::expf(-data_ptr[i])));
  }
 
  //output_ptr                          = data_ptr;
  
  neural_view output_view; 
  output_view.tensor.tensor_data      = input_view.tensor.tensor_data; 
  output_view.tensor.shape.ndim       = 2; 
  output_view.tensor.shape.dims[0]    = batch_size; 
  output_view.tensor.shape.dims[1]    = output_features; 
  output_view.tensor.shape.strides[0] = output_features; 
  output_view.tensor.shape.strides[1] = 1;
  
  return output_view; 
}

void neural::relu::init   (alloc_pool &persistent_arena){}
void neural::sigmoid::init(alloc_pool &persistent_arena){}

void neural::nn::add_linear(size_t input, size_t output){
  layers.push_back(std::make_unique<linear>(input, output)); 
}

void neural::nn::add_relu(){
  layers.push_back(std::make_unique<relu>()); 
}

void neural::nn::add_sigmoid(){
  layers.push_back(std::make_unique<sigmoid>()); 
}

neural::neural_view neural::nn::forward(const neural_view &layer_input_tensor, alloc_pool &arena){
  neural::neural_view current = layer_input_tensor;  
  for(size_t i = 0; i < layers.size(); ++i){
   current = layers[i]->forward(current, arena);
  }
  return current; 
}

void neural::nn::init(alloc_pool &persistent_arena){
  for(auto& layer : layers){
    layer->init(persistent_arena); 
  }
}

size_t neural::nn::mem_reqs(){
  size_t total_size = 0;
  for(auto &layer : layers){
    if(auto *weight = layer->get_weights()){
      total_size += weight->input_features * weight->output_features; 
      total_size += weight->output_features; 
    }
  }
  return total_size * sizeof(float); 
}

float *neural::nn::save_weights(){
  size_t total_size  = this->mem_reqs();
  float *all_weights = new float[total_size]; 
  
  size_t offset = 0; 
  for(auto &layer : layers){
    if(auto *weight = layer->get_weights()){
      size_t count = weight->input_features * weight->output_features; 
      std::memcpy(all_weights + offset, weight->weights.tensor.tensor_data, count * sizeof(float)); 
      offset += count; 
    }
  }
  return all_weights; 
}


uint64_t neural::nn::nanos() {
  struct timespec start;

  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}
