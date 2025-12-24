#include "../../include/attention_core/attention.hpp"

atten::attention::attention(size_t embedded_dim, size_t num_heads) 
: 
  embedded_dim(embedded_dim), 
  num_heads(num_heads) 
{}

void atten::attention::init(atten_pool &persistent_arena){
  size_t d_k = embedded_dim / num_heads; 

  float *ptr_q = persistent_arena.arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_queries.tensor_data   = ptr_q; 
  weights_data.w_queries.shape.dims[0] = embedded_dim; 
  weights_data.w_queries.shape.dims[1] = d_k; 
  weights_data.w_queries.shape.ndim    = 2;

 float *ptr_k = persistent_arena.arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_keys.tensor_data   = ptr_k; 
  weights_data.w_keys.shape.dims[0] = embedded_dim; 
  weights_data.w_keys.shape.dims[1] = d_k;
  weights_data.w_keys.shape.ndim    = 2;

 float *ptr_v = persistent_arena.arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_values.tensor_data   = ptr_v; 
  weights_data.w_values.shape.dims[0] = embedded_dim; 
  weights_data.w_values.shape.dims[1] = d_k; 
  weights_data.w_values.shape.ndim    = 2;
  
  float *ptr_o = persistent_arena.arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_output.tensor_data   = ptr_o; 
  weights_data.w_output.shape.dims[0] = embedded_dim; 
  weights_data.w_output.shape.dims[1] = d_k; 
  weights_data.w_output.shape.ndim    = 2;
}

void atten::attention::load_weights(float *w_q, float *w_k, float *w_v, float *w_o){
  size_t d_k = embedded_dim / num_heads; 
  std::memcpy(weights_data.w_queries.tensor_data, w_q, embedded_dim * d_k * sizeof(float));
  std::memcpy(weights_data.w_values.tensor_data , w_v, embedded_dim * d_k * sizeof(float));
  std::memcpy(weights_data.w_keys.tensor_data   , w_k, embedded_dim * d_k * sizeof(float));
  std::memcpy(weights_data.w_output.tensor_data , w_o, embedded_dim * d_k * sizeof(float));
}

tens::tensor atten::attention::forward(tens::tensor &input_tensor, atten_pool &alloc_pool){
  size_t sequence_length = input_tensor.shape.dims[0]; 
  size_t input_features  = input_tensor.shape.dims[1]; 
  size_t head_dim        = embedded_dim / num_heads;

  float *output_ptr_q       = alloc_pool.arena.nn_alloc<float>(sequence_length * head_dim        ); 
  float *output_ptr_k       = alloc_pool.arena.nn_alloc<float>(sequence_length * head_dim        );
  float *output_ptr_v       = alloc_pool.arena.nn_alloc<float>(sequence_length * head_dim        );
  float *output_ptr_scores  = alloc_pool.arena.nn_alloc<float>(sequence_length * sequence_length ); 
  float *output_ptr_outputs = alloc_pool.arena.nn_alloc<float>(sequence_length * head_dim        );
  float *output_ptr_final   = alloc_pool.arena.nn_alloc<float>(sequence_length * head_dim        ); 

  level3::mat_ops_view input_view {
    .row_view          = sequence_length, 
    .col_view          = input_features, 
    .leading_dimension = input_features, 
    .data_view         = input_tensor.tensor_data
  };
    
  level3::mat_ops_view wq_view {
    .row_view          = weights_data.w_queries.shape.dims[0], 
    .col_view          = weights_data.w_queries.shape.dims[1],
    .leading_dimension = weights_data.w_queries.shape.dims[1], 
    .data_view         = weights_data.w_queries.tensor_data 
  }; 
  
  level3::mat_ops_view wk_view {
      .row_view          = weights_data.w_keys.shape.dims[0], 
      .col_view          = weights_data.w_keys.shape.dims[1],
      .leading_dimension = weights_data.w_keys.shape.dims[1], 
      .data_view         = weights_data.w_keys.tensor_data 
    }; 
  
  level3::mat_ops_view wv_view {
      .row_view          = weights_data.w_values.shape.dims[0], 
      .col_view          = weights_data.w_values.shape.dims[1],
      .leading_dimension = weights_data.w_values.shape.dims[1], 
      .data_view         = weights_data.w_values.tensor_data 
    }; 
  
  level3::mat_ops_view wo_view {
    .row_view          = weights_data.w_output.shape.dims[0], 
    .col_view          = weights_data.w_output.shape.dims[1], 
    .leading_dimension = weights_data.w_output.shape.dims[1], 
    .data_view         = weights_data.w_output.tensor_data
  }; 

  level3::mat_ops_view Q {
    .row_view          = sequence_length, 
    .col_view          = head_dim, 
    .leading_dimension = head_dim, 
    .data_view         = output_ptr_q
  };
  
  level3::mat_ops_view K {
    .row_view          = sequence_length, 
    .col_view          = head_dim, 
    .leading_dimension = head_dim, 
    .data_view         = output_ptr_k
  };

  level3::mat_ops_view V {
    .row_view          = sequence_length, 
    .col_view          = head_dim, 
    .leading_dimension = head_dim, 
    .data_view         = output_ptr_v
  };
  
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, input_view, wq_view, 1.0f, 0.0f, Q); 
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, input_view, wk_view, 1.0f, 0.0f, K);
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, input_view, wv_view, 1.0f, 0.0f, V);
  
  level3::mat_ops_view scores {
    .row_view          = sequence_length, 
    .col_view          = sequence_length,
    .leading_dimension = sequence_length, 
    .data_view         = output_ptr_scores
  };
  
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::transpose, Q, K, 1.0f, 0.0f, scores); 
  
  float attn_scale = 1 / std::sqrt(static_cast<float>(head_dim)); 
  for(size_t i = 0; i < sequence_length * sequence_length; ++i){
    scores.data_view[i] *= attn_scale;
  }
  
  auto weights = level3::blas::softmax(scores);
  
  level3::mat_ops_view attn_output_view {
    .row_view          = sequence_length, 
    .col_view          = head_dim, 
    .leading_dimension = head_dim, 
    .data_view         = output_ptr_outputs
  };
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, weights, V, 1.0f, 0.0f, attn_output_view);
  
  level3::mat_ops_view final_view {
    .row_view          = sequence_length, 
    .col_view          = head_dim, 
    .leading_dimension = head_dim, 
    .data_view         = output_ptr_final
  };
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, attn_output_view, wo_view, 1.0f, 0.0f, final_view); 
  
  tens::tensor output_tensor; 
  output_tensor.shape.dims[0]    = sequence_length; 
  output_tensor.shape.dims[1]    = head_dim; 
  output_tensor.shape.strides[0] = head_dim;
  output_tensor.shape.strides[1] = 1; 
  output_tensor.shape.ndim       = 2; 
  output_tensor.tensor_data      = output_ptr_final; 

  return output_tensor; 
} 





