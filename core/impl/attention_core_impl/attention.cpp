#include "attention_core/attention.hpp"

atten::attention::attention(size_t embedded_dim, size_t num_heads) 
: 
  embedded_dim(embedded_dim), 
  num_heads(num_heads) 
{}

void atten::attention::init(memory::neural_arena &persistent_arena){
  size_t d_k = embedded_dim / num_heads; 

  float *ptr_q = persistent_arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_queries.tensor_data   = ptr_q; 
  weights_data.w_queries.shape.dims[0] = embedded_dim; 
  weights_data.w_queries.shape.dims[1] = d_k; 
  weights_data.w_queries.shape.ndim    = 2;

 float *ptr_k = persistent_arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_keys.tensor_data   = ptr_k; 
  weights_data.w_keys.shape.dims[0] = embedded_dim; 
  weights_data.w_keys.shape.dims[1] = d_k;
  weights_data.w_keys.shape.ndim    = 2;

 float *ptr_v = persistent_arena.nn_alloc<float>(embedded_dim * d_k); 
  weights_data.w_values.tensor_data   = ptr_v; 
  weights_data.w_values.shape.dims[0] = embedded_dim; 
  weights_data.w_values.shape.dims[1] = d_k; 
  weights_data.w_values.shape.ndim    = 2;
  
  float *ptr_o = persistent_arena.nn_alloc<float>(embedded_dim * d_k); 
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

tens::tensor atten::attention::forward(tens::tensor &input_tensor, memory::neural_arena &alloc_pool){
  size_t sequence_length = input_tensor.shape.dims[0]; 
  size_t input_features  = input_tensor.shape.dims[1]; 
  size_t head_dim        = embedded_dim / num_heads;

  float *output_ptr_q       = alloc_pool.nn_alloc<float>( sequence_length * head_dim        ); 
  float *output_ptr_k       = alloc_pool.nn_alloc<float>( sequence_length * head_dim        );
  float *output_ptr_v       = alloc_pool.nn_alloc<float>( sequence_length * head_dim        );
  float *output_ptr_scores  = alloc_pool.nn_alloc<float>( sequence_length * sequence_length ); 
  float *output_ptr_outputs = alloc_pool.nn_alloc<float>( sequence_length * head_dim        );
  float *output_ptr_final   = alloc_pool.nn_alloc<float>( sequence_length * head_dim        ); 

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

atten::multi_head_attention::multi_head_attention(size_t embedded_dim, size_t num_heads) 
: 
  embedded_dim(embedded_dim), 
  num_heads(num_heads) 
{}


void atten::multi_head_attention::init(memory::neural_arena &persistent_arena){
  size_t weights_size = embedded_dim * embedded_dim; 
  
  weights_data.w_queries.tensor_data = persistent_arena.nn_alloc<float>(weights_size);
  weights_data.w_queries.shape.ndim = 2;
  weights_data.w_queries.shape.dims[0] = embedded_dim;
  weights_data.w_queries.shape.dims[1] = embedded_dim;
  weights_data.w_queries.shape.strides[0] = embedded_dim;
  weights_data.w_queries.shape.strides[1] = 1;
  
  weights_data.w_keys.tensor_data = persistent_arena.nn_alloc<float>(weights_size);
  weights_data.w_keys.shape.ndim = 2;
  weights_data.w_keys.shape.dims[0] = embedded_dim;
  weights_data.w_keys.shape.dims[1] = embedded_dim;
  weights_data.w_keys.shape.strides[0] = embedded_dim;
  weights_data.w_keys.shape.strides[1] = 1;
  
  weights_data.w_values.tensor_data = persistent_arena.nn_alloc<float>(weights_size);
  weights_data.w_values.shape.ndim = 2;
  weights_data.w_values.shape.dims[0] = embedded_dim;
  weights_data.w_values.shape.dims[1] = embedded_dim;
  weights_data.w_values.shape.strides[0] = embedded_dim;
  weights_data.w_values.shape.strides[1] = 1;
  
  weights_data.w_output.tensor_data = persistent_arena.nn_alloc<float>(weights_size);
  weights_data.w_output.shape.ndim = 2;
  weights_data.w_output.shape.dims[0] = embedded_dim;
  weights_data.w_output.shape.dims[1] = embedded_dim;
  weights_data.w_output.shape.strides[0] = embedded_dim;
  weights_data.w_output.shape.strides[1] = 1;
  
  weights_data.b_queries.tensor_data = persistent_arena.nn_alloc<float>(embedded_dim);
  weights_data.b_queries.shape.ndim = 1;
  weights_data.b_queries.shape.dims[0] = embedded_dim;
  weights_data.b_queries.shape.strides[0] = 1;
  
  weights_data.b_keys.tensor_data = persistent_arena.nn_alloc<float>(embedded_dim);
  weights_data.b_keys.shape.ndim = 1;
  weights_data.b_keys.shape.dims[0] = embedded_dim;
  weights_data.b_keys.shape.strides[0] = 1;
  
  weights_data.b_values.tensor_data = persistent_arena.nn_alloc<float>(embedded_dim);
  weights_data.b_values.shape.ndim = 1;
  weights_data.b_values.shape.dims[0] = embedded_dim;
  weights_data.b_values.shape.strides[0] = 1;
  
  weights_data.b_output.tensor_data = persistent_arena.nn_alloc<float>(embedded_dim);
  weights_data.b_output.shape.ndim = 1;
  weights_data.b_output.shape.dims[0] = embedded_dim;
  weights_data.b_output.shape.strides[0] = 1;
}

void atten::multi_head_attention::load_weights(float *w_q, float *w_k, float *w_v, float *w_o, float *b_q, float *b_k, float *b_v, float *b_o){
  size_t weights_size = embedded_dim * embedded_dim; 
  
  std::memcpy(weights_data.w_queries.tensor_data, w_q, weights_size * sizeof(float));
  std::memcpy(weights_data.w_values.tensor_data , w_v, weights_size * sizeof(float));
  std::memcpy(weights_data.w_keys.tensor_data   , w_k, weights_size * sizeof(float));
  std::memcpy(weights_data.w_output.tensor_data , w_o, weights_size * sizeof(float));

  std::memcpy(weights_data.b_queries.tensor_data, b_q, embedded_dim * sizeof(float));
  std::memcpy(weights_data.b_keys.tensor_data,    b_k, embedded_dim * sizeof(float));
  std::memcpy(weights_data.b_values.tensor_data,  b_v, embedded_dim * sizeof(float));
  std::memcpy(weights_data.b_output.tensor_data,  b_o, embedded_dim * sizeof(float));
}


tens::tensor atten::multi_head_attention::forward(tens::tensor &input_tensor, memory::neural_arena &alloc_pool) {
  size_t sequence_length = input_tensor.shape.dims[0]; 
  size_t embed_dim       = input_tensor.shape.dims[1]; 
  size_t head_dim        = embed_dim / num_heads;

  float *output_ptr_q       = alloc_pool.nn_alloc<float>(sequence_length * embed_dim); 
  float *output_ptr_k       = alloc_pool.nn_alloc<float>(sequence_length * embed_dim);
  float *output_ptr_v       = alloc_pool.nn_alloc<float>(sequence_length * embed_dim);
  float *output_ptr_scores  = alloc_pool.nn_alloc<float>(num_heads * sequence_length * sequence_length); 
  float *output_ptr_outputs = alloc_pool.nn_alloc<float>(sequence_length * embed_dim);
  float *output_ptr_final   = alloc_pool.nn_alloc<float>(sequence_length * embed_dim); 

  level3::mat_ops_view input_view { 
    .row_view          = sequence_length, 
    .col_view          = embed_dim, 
    .leading_dimension = embed_dim, 
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
    .col_view          = embed_dim, 
    .leading_dimension = embed_dim, 
    .data_view         = output_ptr_q 
  };

  level3::mat_ops_view K { 
    .row_view          = sequence_length, 
    .col_view          = embed_dim, 
    .leading_dimension = embed_dim, 
    .data_view         = output_ptr_k 
  }; 
  
  level3::mat_ops_view V { 
    .row_view          = sequence_length, 
    .col_view          = embed_dim, 
    .leading_dimension = embed_dim, 
    .data_view         = output_ptr_v 
  }; 
  
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, input_view, wq_view, 1.0f, 0.0f, Q);
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, input_view, wk_view, 1.0f, 0.0f, K);
  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, input_view, wv_view, 1.0f, 0.0f, V);

  for (size_t s = 0; s < sequence_length; s++){
    for (size_t d = 0; d < embed_dim; d++){
      Q.data_view[s * embed_dim + d] += weights_data.b_queries.tensor_data[d];
      K.data_view[s * embed_dim + d] += weights_data.b_keys.tensor_data[d];
      V.data_view[s * embed_dim + d] += weights_data.b_values.tensor_data[d];
    }
  }
  
  if (sequence_length == 8) { 
    std::printf("[DEBUG] W_Q[0][0:3]: %.6f %.6f %.6f\n",
                weights_data.w_queries.tensor_data[0],
                weights_data.w_queries.tensor_data[1],
                weights_data.w_queries.tensor_data[2]);
  }

  if (sequence_length == 1) {
    std::printf("[DEBUG] Q[0][0:3]: %.6f %.6f %.6f\n", 
                Q.data_view[0], Q.data_view[1], Q.data_view[2]);
    std::printf("[DEBUG] K[0][0:3]: %.6f %.6f %.6f\n", 
                K.data_view[0], K.data_view[1], K.data_view[2]);
    std::printf("[DEBUG] V[0][0:3]: %.6f %.6f %.6f\n", 
                V.data_view[0], V.data_view[1], V.data_view[2]);
  }
 
  tens::tensor k_tensor_wrapper;
  k_tensor_wrapper.shape.ndim = 2;
  k_tensor_wrapper.shape.dims[0] = sequence_length;
  k_tensor_wrapper.shape.dims[1] = embed_dim;
  k_tensor_wrapper.tensor_data = output_ptr_k;

tens::tensor K_Transposed = tens::ops::cpu_transpose_avx512(k_tensor_wrapper, alloc_pool);
  
  float scale = 1.0f / std::sqrt((float)head_dim);
  float minus_inf = -1e9f;

  for(size_t head = 0; head < num_heads; ++head){
    size_t head_offset_flat = head * head_dim; 

    level3::mat_ops_view q_head {
        .row_view          = sequence_length, 
        .col_view          = head_dim, 
        .leading_dimension = embed_dim, 
        .data_view         = Q.data_view + head_offset_flat
    };

    size_t k_offset_transposed = head * head_dim * sequence_length;
    
    level3::mat_ops_view k_head_T {
        .row_view          = head_dim, 
        .col_view          = sequence_length, 
        .leading_dimension = sequence_length,
        .data_view         = K_Transposed.tensor_data + k_offset_transposed
    };

    level3::mat_ops_view scores_head {
        .row_view          = sequence_length, 
        .col_view          = sequence_length, 
        .leading_dimension = sequence_length, 
        .data_view         = output_ptr_scores + head * sequence_length * sequence_length
    };
        
    level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, q_head, k_head_T, 1.0f, 0.0f, scores_head);
          
    for (size_t i = 0; i < sequence_length; i++){
      for (size_t j = 0; j < sequence_length; j++){
        size_t idx = i * sequence_length + j;
        if (j > i) {
          scores_head.data_view[idx] = minus_inf;
        } else {
          scores_head.data_view[idx] *= scale;
        }
      }
    }

    auto weights_head = level3::blas::softmax(scores_head);

      
    level3::mat_ops_view v_head {
        .row_view          = sequence_length, 
        .col_view          = head_dim, 
        .leading_dimension = embed_dim, 
        .data_view         = V.data_view + head_offset_flat
    };
    
    float* tmp_out_ptr = alloc_pool.nn_alloc<float>(sequence_length * head_dim);

    level3::mat_ops_view atten_head_output_tmp {
        .row_view          = sequence_length, 
        .col_view          = head_dim, 
        .leading_dimension = head_dim, 
        .data_view         = tmp_out_ptr
    };

    level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, weights_head, v_head, 1.0f, 0.0f, atten_head_output_tmp); 

    for(size_t s = 0; s < sequence_length; s++) {
        float* src = tmp_out_ptr + (s * head_dim);
        float* dst = output_ptr_outputs + (s * embed_dim) + head_offset_flat;
        std::memcpy(dst, src, head_dim * sizeof(float));
    }
  }

  level3::mat_ops_view atten_output_view { 
    .row_view          = sequence_length, 
    .col_view          = embed_dim, 
    .leading_dimension = embed_dim, 
    .data_view         = output_ptr_outputs 
  };

  level3::mat_ops_view final_view { 
    .row_view          = sequence_length, 
    .col_view          = embed_dim, 
    .leading_dimension = embed_dim, 
    .data_view         = output_ptr_final 
  };

  level3::blas::crush_gemm(level3::transpose_gemm::no_transpose, level3::transpose_gemm::no_transpose, atten_output_view, wo_view, 1.0f, 0.0f, final_view);
  
  for (size_t s = 0; s < sequence_length; s++) {
    for (size_t d = 0; d < embed_dim; d++) {
      final_view.data_view[s * embed_dim + d] += weights_data.b_output.tensor_data[d];
    } 
  }
    
  tens::tensor output_tensor; 
  output_tensor.shape.dims[0]    = sequence_length; 
  output_tensor.shape.dims[1]    = embed_dim; 
  output_tensor.shape.strides[0] = embed_dim;
  output_tensor.shape.strides[1] = 1; 
  output_tensor.shape.ndim       = 2; 
  output_tensor.tensor_data      = output_ptr_final; 

  return output_tensor; 
}
