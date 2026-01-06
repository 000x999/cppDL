#include "model_core/gpt2.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace gpt2{
void debug_tensor(const char* name, const tens::tensor& t) {
    float min_val = t.tensor_data[0];
    float max_val = t.tensor_data[0];
    float sum = 0;
    bool has_nan = false;
    bool has_inf = false;
    
    size_t numel = t.shape.numel();
    for (size_t i = 0; i < numel; i++) {
        float v = t.tensor_data[i];
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    
    std::printf("[%s] shape=[", name);
    for (int i = 0; i < t.shape.ndim; i++) {
        std::printf("%zu%s", t.shape.dims[i], i < t.shape.ndim - 1 ? "," : "");
    }
    std::printf("] min=%.6f max=%.6f mean=%.6f nan=%d inf=%d\n",
                min_val, max_val, sum / numel, has_nan, has_inf);
}

tens::tensor load_tensor(safetensor::safetensor_file *sf, const char *name, memory::neural_arena &alloc) {
  safetensor::tensor_entry* entry = safetensor::find_entry(sf, name);
  if (!entry) {
    std::printf("ERROR: tensor '%s' not found\n", name);
    tens::tensor empty;
    empty.tensor_data = nullptr;
    return empty;
  }
  
  tens::tensor t;
  t.shape.ndim = entry->ndim;
  
  size_t numel = 1;
  for (int i = 0; i < entry->ndim; i++) {
    t.shape.dims[i] = entry->shape[i];
    t.shape.strides[i] = entry->strides[i];
    numel *= entry->shape[i];
  }
  
  t.tensor_data = alloc.nn_alloc<float>(numel);
  float* src = reinterpret_cast<float*>(sf->tensor_data_start + entry->data_offset);
  std::memcpy(t.tensor_data, src, numel * sizeof(float));
  
  return t;
}

tens::tensor matmul(const tens::tensor &a, const tens::tensor &b, memory::neural_arena &pool){
  size_t M = a.shape.dims[0];
  size_t K = a.shape.dims[1];
  size_t N = b.shape.dims[1]; 
  
  tens::tensor out;
  out.shape.ndim = 2;
  out.shape.dims[0] = M;
  out.shape.dims[1] = N;
  out.shape.strides[0] = N;
  out.shape.strides[1] = 1;
  out.tensor_data = pool.nn_alloc<float>(M * N);
  
  std::memset(out.tensor_data, 0, M * N * sizeof(float));
  
  level3::mat_ops_view view_a {
    .row_view = M,
    .col_view = K,
    .leading_dimension = a.shape.strides[0],
    .data_view = a.tensor_data
  };
  
  level3::mat_ops_view view_b {
    .row_view = b.shape.dims[0],
    .col_view = b.shape.dims[1],
    .leading_dimension = b.shape.strides[0],
    .data_view = b.tensor_data
  };
  
  level3::mat_ops_view view_out {
    .row_view = M,
    .col_view = N,
    .leading_dimension = N,
    .data_view = out.tensor_data
  };
  
  level3::blas::crush_gemm(
    level3::transpose_gemm::no_transpose,
    level3::transpose_gemm::no_transpose,
    view_a,
    view_b,
    1.0f,
    0.0f,
    view_out
  );
  
  return out;
}

/*
tens::tensor linear(const tens::tensor &x, const tens::tensor &weight, const tens::tensor &bias, memory::neural_arena &pool) {
size_t seq_len = x.shape.dims[0];
  size_t in_features = x.shape.dims[1];
  
  if (weight.shape.dims[0] != in_features) {
      std::printf("[FATAL] Linear Dimension Mismatch! Input Col: %zu, Weight Row: %zu\n", 
                  in_features, weight.shape.dims[0]);
      std::printf("       Hint: Did you forget to transpose the PyTorch weights?\n");
      std::exit(1);
  }

  size_t out_features = weight.shape.dims[1]; 
  
  tens::tensor out;
  out.shape.ndim = 2;
  out.shape.dims[0] = seq_len;
  out.shape.dims[1] = out_features;
  out.shape.strides[0] = out_features;
  out.shape.strides[1] = 1;
  out.tensor_data = pool.nn_alloc<float>(seq_len * out_features);
  
  std::memset(out.tensor_data, 0, seq_len * out_features * sizeof(float));
  
  level3::mat_ops_view view_x {
    .row_view = seq_len,
    .col_view = in_features,
    .leading_dimension = in_features,
    .data_view = x.tensor_data
  };
  
  level3::mat_ops_view view_w {
    .row_view = in_features,
    .col_view = out_features,
    .leading_dimension = out_features, 
    .data_view = weight.tensor_data
  };
  
  level3::mat_ops_view view_out {
    .row_view = seq_len,
    .col_view = out_features,
    .leading_dimension = out_features,
    .data_view = out.tensor_data
  };
  
  level3::blas::crush_gemm(
    level3::transpose_gemm::no_transpose,
    level3::transpose_gemm::no_transpose, 
    view_x,
    view_w,
    1.0f,
    0.0f,
    view_out
  );
  
  for (size_t i = 0; i < seq_len; i++) {
    for (size_t j = 0; j < out_features; j++) {
      out.tensor_data[i * out_features + j] += bias.tensor_data[j];
    }
  }
  return out;
}
*/

tens::tensor linear(const tens::tensor& input, const tens::tensor& weight, const tens::tensor& bias, memory::neural_arena& pool) {
  tens::tensor output = matmul(input, weight, pool); 

  size_t seq_len    = output.shape.dims[0];
  size_t output_dim = output.shape.dims[1]; 

  if (bias.shape.numel() != output_dim) {
     printf("FATAL: Linear bias dim %zu != output dim %zu\n", bias.shape.numel(), output_dim);
     exit(1);
  }

  for (size_t i = 0; i < seq_len; i++) {
    float* out_row  = output.tensor_data + (i * output_dim);
    float* bias_ptr = bias.tensor_data;
    size_t j = 0;
    
    for (; j + 15 < output_dim; j += 16) {
      __m512 v_out  = _mm512_loadu_ps(out_row + j);
      __m512 v_bias = _mm512_loadu_ps(bias_ptr + j);
      _mm512_storeu_ps(out_row + j, _mm512_add_ps(v_out, v_bias));
    }
    for (; j < output_dim; j++) {
      out_row[j] += bias_ptr[j];
    }
  }

  return output;
}

void init_model(model* m) {
  m->initialized        = false;
  m->cfg.vocab_size     = 0;
  m->cfg.embed_dim      = 0;
  m->cfg.num_heads      = 0;
  m->cfg.num_layers     = 0;
  m->cfg.max_seq_len    = 0;
  m->cfg.layer_norm_eps = 1e-5f;
  
  m->wte.tensor_data         = nullptr;
  m->wpe.tensor_data         = nullptr;
  m->ln_f_weight.tensor_data = nullptr;
  m->ln_f_bias.tensor_data   = nullptr;
  
  for (size_t i = 0; i < MAX_LAYERS; i++) {
    m->attentions[i]                         = nullptr;
    m->atten_pools[i]                        = nullptr;
    m->blocks[i].ln1_weight.tensor_data      = nullptr;
    m->blocks[i].ln1_bias.tensor_data        = nullptr;
    m->blocks[i].ln2_weight.tensor_data      = nullptr;
    m->blocks[i].ln2_bias.tensor_data        = nullptr;
    m->blocks[i].ffn_fc_weight.tensor_data   = nullptr;
    m->blocks[i].ffn_fc_bias.tensor_data     = nullptr;
    m->blocks[i].ffn_proj_weight.tensor_data = nullptr;
    m->blocks[i].ffn_proj_bias.tensor_data   = nullptr;
  }
}

bool load_model(model *m, const char *path, memory::neural_arena &alloc) {
  init_model(m);
  
  safetensor::safetensor_file sf;
  if (!safetensor::load_safetensor(path, &sf)) {
    std::printf("ERROR: failed to load safetensor file\n");
    return false;
  }

  std::printf("Loaded tensors:\n");
  safetensor::print_entries(&sf);
  std::printf("\n");
  
  m->wte = load_tensor(&sf, "wte.weight", alloc);
  m->wpe = load_tensor(&sf, "wpe.weight", alloc);
  
  if (m->wte.tensor_data) {
    m->cfg.vocab_size = m->wte.shape.dims[0];
    m->cfg.embed_dim  = m->wte.shape.dims[1]; 
  }
  if (m->wpe.tensor_data) {
    m->cfg.max_seq_len = m->wpe.shape.dims[0];
  }
  m->cfg.layer_norm_eps = 1e-5f;
    
  m->cfg.num_layers = 0;
  char name[safetensor::MAX_NAME_LEN];
  
  for (size_t i = 0; i < MAX_LAYERS; i++) {
    std::snprintf(name, sizeof(name), "h.%zu.ln_1.weight", i);
    if (safetensor::find_entry(&sf, name)) {
      m->cfg.num_layers++;
    } else {
      break;
    }
  }
  m->cfg.num_heads = m->cfg.embed_dim / 64;
  
  std::printf("Detected config: vocab=%zu, embed=%zu, layers=%zu, heads=%zu\n",
               m->cfg.vocab_size, m->cfg.embed_dim, m->cfg.num_layers, m->cfg.num_heads);

  m->wte_T.shape.ndim = 2;
  m->wte_T.shape.dims[0] = m->cfg.embed_dim;
  m->wte_T.shape.dims[1] = m->cfg.vocab_size;
  m->wte_T.shape.strides[0] = m->cfg.vocab_size;
  m->wte_T.shape.strides[1] = 1;
  m->wte_T.tensor_data = alloc.nn_alloc<float>(m->cfg.embed_dim * m->cfg.vocab_size);

  size_t vocab = m->cfg.vocab_size;
  size_t embed = m->cfg.embed_dim;

  for (size_t v = 0; v < vocab; v++) {
      for (size_t e = 0; e < embed; e++) {
          float val = m->wte.tensor_data[v * embed + e];
          m->wte_T.tensor_data[e * vocab + v] = val;
      }
  }

  size_t embed_dim = m->cfg.embed_dim;
  size_t atten_arena_size = MAX_SEQ_LEN * embed_dim * 16 * sizeof(float);
 
  for (size_t i = 0; i < m->cfg.num_layers; i++) {
    transformer_block* b = &m->blocks[i];
    
    std::snprintf(name, sizeof(name), "h.%zu.ln_1.weight", i);
    b->ln1_weight = load_tensor(&sf, name, alloc);
    std::snprintf(name, sizeof(name), "h.%zu.ln_1.bias", i);
    b->ln1_bias = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.attn.c_attn.weight", i);
    tens::tensor qkv_weight = load_tensor(&sf, name, alloc);
  
    std::snprintf(name, sizeof(name), "h.%zu.attn.c_attn.bias", i);
    tens::tensor qkv_bias = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.attn.c_proj.weight", i);
    tens::tensor attn_proj_weight = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.attn.c_proj.bias", i);
    tens::tensor attn_proj_bias = load_tensor(&sf, name, alloc);
    
    float* w_q_buf = alloc.nn_alloc<float>(embed_dim * embed_dim);
    float* w_k_buf = alloc.nn_alloc<float>(embed_dim * embed_dim);
    float* w_v_buf = alloc.nn_alloc<float>(embed_dim * embed_dim);

   for (size_t row = 0; row < embed_dim; row++) {
    size_t src_row_offset = row * (3 * embed_dim);
    for (size_t col = 0; col < embed_dim; col++) {
      w_q_buf[row * embed_dim + col] = qkv_weight.tensor_data[src_row_offset + col];
      w_k_buf[row * embed_dim + col] = qkv_weight.tensor_data[src_row_offset + embed_dim + col];
      w_v_buf[row * embed_dim + col] = qkv_weight.tensor_data[src_row_offset + 2 * embed_dim + col];
    }
  } 
    
    float* b_q = qkv_bias.tensor_data;
    float* b_k = qkv_bias.tensor_data + embed_dim;
    float* b_v = qkv_bias.tensor_data + 2 * embed_dim;
    float* w_o = attn_proj_weight.tensor_data; 
    float* b_o = attn_proj_bias.tensor_data;
    
    void* pool_mem  = alloc.nn_alloc<char>(sizeof(memory::neural_arena));
    void* atten_mem = alloc.nn_alloc<char>(sizeof(atten::multi_head_attention));
    
    m->atten_pools[i] = new (pool_mem) memory::neural_arena(atten_arena_size);
    m->attentions[i]  = new (atten_mem) atten::multi_head_attention(embed_dim, m->cfg.num_heads);
    m->attentions[i]->init(alloc);
    m->attentions[i]->load_weights(w_q_buf, w_k_buf, w_v_buf, w_o, b_q, b_k, b_v, b_o);
    
    std::snprintf(name, sizeof(name), "h.%zu.ln_2.weight", i);
    b->ln2_weight = load_tensor(&sf, name, alloc);
    std::snprintf(name, sizeof(name), "h.%zu.ln_2.bias", i);
    b->ln2_bias = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.weight", i);
    b->ffn_fc_weight = load_tensor(&sf, name, alloc); 
    
    std::snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.bias", i);
    b->ffn_fc_bias = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.weight", i);
    b->ffn_proj_weight = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.bias", i);
    b->ffn_proj_bias = load_tensor(&sf, name, alloc);
  }
    
  m->ln_f_weight = load_tensor(&sf, "ln_f.weight", alloc);
  m->ln_f_bias = load_tensor(&sf, "ln_f.bias", alloc);
  
  safetensor::free_safetensor(&sf);
  m->initialized = true;
  std::printf("Model loaded successfully\n");
  return true;
}

/*
tens::tensor forward(model *m, const tens::tensor &tokens, tens::tensor_pool &pool) {
    for (size_t i = 0; i < m->cfg.num_layers; i++) {
      m->atten_pools[i]->arena.nn_reset();
    }
    static int call_count = 0;
    bool debug = (call_count == 0); 
    call_count++;
    
    size_t seq_len = tokens.shape.dims[0];
    size_t embed_dim = m->cfg.embed_dim;
    
    tens::tensor x = tens::ops::embedding(m->wte, tokens, pool);
    
    tens::tensor pos_emb;
    pos_emb.shape.ndim = 2;
    pos_emb.shape.dims[0] = seq_len;
    pos_emb.shape.dims[1] = embed_dim;
    pos_emb.shape.strides[0] = embed_dim;
    pos_emb.shape.strides[1] = 1;
    pos_emb.tensor_data = m->wpe.tensor_data;
    
    x = tens::ops::add(x, pos_emb, pool);
    
    for (size_t i = 0; i < m->cfg.num_layers; i++) {
      transformer_block* b = &m->blocks[i];
      tens::tensor residual = x;
      
      x = tens::ops::layer_norm(x, b->ln1_weight, b->ln1_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
      
      x = m->attentions[i]->forward(x, *m->atten_pools[i]);
      
      x = tens::ops::add(residual, x, pool);
      residual = x;
      
      x = tens::ops::layer_norm(x, b->ln2_weight, b->ln2_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
      
      x = linear(x, b->ffn_fc_weight, b->ffn_fc_bias, pool);
      
      x = tens::ops::gelu(x, pool);
      x = linear(x, b->ffn_proj_weight, b->ffn_proj_bias, pool);
      x = tens::ops::add(residual, x, pool);
    }
    
    x = tens::ops::layer_norm(x, m->ln_f_weight, m->ln_f_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);

    tens::tensor wte_transposed = m->wte_T;
    
    if (wte_transposed.shape.dims[0] != embed_dim) {
      if (debug) printf("[WARN] wte_T not pre-transposed. Transposing now (Slow!)...\n");
      wte_transposed = tens::ops::cpu_transpose_avx512(m->wte, pool);
    }

    tens::tensor logits;
    logits.shape.ndim = 2;
    logits.shape.dims[0] = seq_len;
    logits.shape.dims[1] = m->cfg.vocab_size;
    logits.shape.strides[0] = m->cfg.vocab_size;
    logits.shape.strides[1] = 1;
    logits.tensor_data = pool.arena.nn_alloc<float>(seq_len * m->cfg.vocab_size);
    
    std::memset(logits.tensor_data, 0, seq_len * m->cfg.vocab_size * sizeof(float));
    
    level3::mat_ops_view view_x {
        .row_view = seq_len,
        .col_view = embed_dim,
        .leading_dimension = embed_dim,
        .data_view = x.tensor_data
    };
    
    level3::mat_ops_view view_wte_T {
        .row_view = embed_dim,
        .col_view = m->cfg.vocab_size,
        .leading_dimension = m->cfg.vocab_size, 
        .data_view = wte_transposed.tensor_data
    };
    
    level3::mat_ops_view view_logits {
        .row_view = seq_len,
        .col_view = m->cfg.vocab_size,
        .leading_dimension = m->cfg.vocab_size,
        .data_view = logits.tensor_data
    };
   
    level3::blas::crush_gemm(
        level3::transpose_gemm::no_transpose,
        level3::transpose_gemm::no_transpose, 
        view_x,
        view_wte_T,
        1.0f,
        0.0f,
        view_logits
    );
    return logits;
}
*/

tens::tensor forward(model *m, const tens::tensor &tokens, memory::neural_arena &pool) {
    static int call_count = 0;
    bool debug = (call_count == 0); 
    call_count++;
    
    size_t seq_len = tokens.shape.dims[0];
    size_t embed_dim = m->cfg.embed_dim;

    tens::tensor x;
    x.shape.ndim = 2;
    x.shape.dims[0] = seq_len;
    x.shape.dims[1] = embed_dim;
    x.shape.strides[0] = embed_dim;
    x.shape.strides[1] = 1;
    x.tensor_data = pool.nn_alloc<float>(seq_len * embed_dim);

    for (size_t t = 0; t < seq_len; t++) {
        int token_id = (int)tokens.tensor_data[t];
        
        if (token_id < 0 || token_id >= m->cfg.vocab_size) {
             token_id = 50256; 
        }

        float* wte_row = m->wte.tensor_data + (token_id * embed_dim);
        
        float* wpe_row = m->wpe.tensor_data + (t * embed_dim);
        
        float* target_row = x.tensor_data + (t * embed_dim);

        size_t wpe_offset = t * embed_dim; 
        for (size_t j = 0; j < embed_dim; j++) {
            target_row[j] = wte_row[j] + wpe_row[j];
        }
    }

    for (size_t i = 0; i < m->cfg.num_layers; i++) {
        transformer_block* b = &m->blocks[i];
        tens::tensor residual = x;
        
        x = tens::ops::layer_norm(x, b->ln1_weight, b->ln1_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
        
        x = m->attentions[i]->forward(x, *m->atten_pools[i]);
        
        x = tens::ops::add(residual, x, pool);
        residual = x;
        
        x = tens::ops::layer_norm(x, b->ln2_weight, b->ln2_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
        
        x = linear(x, b->ffn_fc_weight, b->ffn_fc_bias, pool);
        x = tens::ops::gelu(x, pool);
        x = linear(x, b->ffn_proj_weight, b->ffn_proj_bias, pool);
        
        x = tens::ops::add(residual, x, pool);
    }
    
    x = tens::ops::layer_norm(x, m->ln_f_weight, m->ln_f_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);

    tens::tensor wte_transposed = m->wte_T;
   
    tens::tensor logits;
    logits.shape.ndim = 2;
    logits.shape.dims[0] = seq_len;
    logits.shape.dims[1] = m->cfg.vocab_size;
    logits.shape.strides[0] = m->cfg.vocab_size;
    logits.shape.strides[1] = 1;
    logits.tensor_data = pool.nn_alloc<float>(seq_len * m->cfg.vocab_size);
    
    level3::mat_ops_view view_x {
        .row_view = seq_len,
        .col_view = embed_dim,
        .leading_dimension = embed_dim,
        .data_view = x.tensor_data
    };
    
    level3::mat_ops_view view_wte_T {
        .row_view = embed_dim,
        .col_view = m->cfg.vocab_size,
        .leading_dimension = m->cfg.vocab_size, 
        .data_view = wte_transposed.tensor_data
    };
    
    level3::mat_ops_view view_logits {
        .row_view = seq_len,
        .col_view = m->cfg.vocab_size,
        .leading_dimension = m->cfg.vocab_size,
        .data_view = logits.tensor_data
    };
    
    level3::blas::crush_gemm(
        level3::transpose_gemm::no_transpose,
        level3::transpose_gemm::no_transpose, 
        view_x,
        view_wte_T,
        1.0f,
        0.0f,
        view_logits
    );
    
    return logits;
}

int argmax(const tens::tensor& logits) {
    size_t vocab_size = logits.shape.dims[1];
    size_t seq_len = logits.shape.dims[0];
    
    float* last_row = logits.tensor_data + (seq_len - 1) * vocab_size;
    
    float top_vals[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
    int top_idxs[5] = {0, 0, 0, 0, 0};
    
    for (size_t i = 0; i < vocab_size; i++) {
        float val = last_row[i];
        for (int k = 0; k < 5; k++) {
            if (val > top_vals[k]) {
                for (int j = 4; j > k; j--) {
                    top_vals[j] = top_vals[j-1];
                    top_idxs[j] = top_idxs[j-1];  
                }
                top_vals[k] = val;
                top_idxs[k] = i;
                break;
            }
        }
    }
    
    for (int k = 0; k < 5; k++) {
      std::printf("%d(%.2f) ", top_idxs[k], top_vals[k]);
    }
    std::printf("\n");
    
    return top_idxs[0];
}

void free_model(model* m) {

  for (size_t i = 0; i < m->cfg.num_layers; i++) {
    if (m->attentions[i]) {
      m->attentions[i]->~multi_head_attention();
      m->attentions[i] = nullptr;
    }
    if (m->atten_pools[i]) {
      m->atten_pools[i]->~neural_arena();
      m->atten_pools[i] = nullptr;
    }
  }
  m->initialized = false;
}

int sample_top_k(float* logits, size_t vocab_size, int k) {
  std::vector<std::pair<int, float>> pairs(vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    pairs[i] = { (int)i, logits[i] };
  }

  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), 
                    [](const auto& a, const auto& b) {
                        return a.second > b.second; 
                    });

  float max_logit = pairs[0].second;
  float sum_exp = 0.0f;
  std::vector<float> probs(k);
  
  for (int i = 0; i < k; i++) {
    probs[i] = std::exp(pairs[i].second - max_logit);
    sum_exp += probs[i];
  }

  static std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, sum_exp);
  float r = dist(rng);
  
  float cum_prob = 0.0f;
  for (int i = 0; i < k; i++) {
    cum_prob += probs[i];
    if (r <= cum_prob) {
      return pairs[i].first;
    }
  }
  
  return pairs[k-1].first; 
}

int sample_top_k_avx512(float* logits, size_t vocab_size, int k, float temperature, memory::neural_arena &pool) {
  Token* top_k = pool.nn_alloc<Token>(k + 1);
  
  for (int i = 0; i < k; ++i) {
    top_k[i] = { -1e30f, -1 };
  }

  float threshold = -1e30f;

  auto insert_candidate = [&](float score, int id) {
    if (score <= threshold) return;

    int i = k - 1;
    while (i >= 0 && score > top_k[i].score) {
      if (i < k - 1) {
        top_k[i + 1] = top_k[i];
      }
      i--;
  }
    if (i + 1 < k) {
      top_k[i + 1] = { score, id };
    }
    
    threshold = top_k[k - 1].score;
  };

  size_t i = 0;
  
  __m512i v_indices_base = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  __m512i v_indices_step = _mm512_set1_epi32(16);
  __m512i v_current_indices = v_indices_base;

  for (; i + 16 <= vocab_size; i += 16) {
    __m512 v_logits = _mm512_loadu_ps(&logits[i]);
    
    __m512 v_threshold = _mm512_set1_ps(threshold);
    
    __mmask16 mask = _mm512_cmp_ps_mask(v_logits, v_threshold, _CMP_GT_OQ);

  if (mask) {
    while (mask) {
      int bit_idx = __builtin_ctz(mask);
      
      float val = logits[i + bit_idx];
      int id = i + bit_idx;
      
      insert_candidate(val, id);
      
      mask &= ~(1 << bit_idx); 
    }
  }
  v_current_indices = _mm512_add_epi32(v_current_indices, v_indices_step);
}

for (; i < vocab_size; ++i) {
  insert_candidate(logits[i], i);
}

float* probs = pool.nn_alloc<float>(k);

float max_logit = top_k[0].score / temperature; 
float sum_exp = 0.0f;

for (int j = 0; j < k; j++) {
  float val = (top_k[j].score / temperature);
    probs[j] = std::exp(val - max_logit);
    sum_exp += probs[j];
  }

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum_exp);
    float r = dist(rng);
    
    float cum_prob = 0.0f;
    for (int j = 0; j < k; j++) {
      cum_prob += probs[j];
      if (r <= cum_prob) {
        return top_k[j].id;
      }
    }
  return top_k[k-1].id;
}
}

