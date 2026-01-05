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

tens::tensor matmul(const tens::tensor &a, const tens::tensor &b, tens::tensor_pool &pool){
  size_t M = a.shape.dims[0];
  size_t K = a.shape.dims[1];
  size_t N = b.shape.dims[1]; 
  
  tens::tensor out;
  out.shape.ndim = 2;
  out.shape.dims[0] = M;
  out.shape.dims[1] = N;
  out.shape.strides[0] = N;
  out.shape.strides[1] = 1;
  out.tensor_data = pool.arena.nn_alloc<float>(M * N);
  
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

tens::tensor linear(const tens::tensor &x, const tens::tensor &weight, const tens::tensor &bias, tens::tensor_pool &pool) {
  size_t seq_len = x.shape.dims[0];
  size_t in_features = x.shape.dims[1];
  size_t out_features = weight.shape.dims[1]; 
  
  tens::tensor out;
  out.shape.ndim = 2;
  out.shape.dims[0] = seq_len;
  out.shape.dims[1] = out_features;
  out.shape.strides[0] = out_features;
  out.shape.strides[1] = 1;
  out.tensor_data = pool.arena.nn_alloc<float>(seq_len * out_features);
  
  std::memset(out.tensor_data, 0, seq_len * out_features * sizeof(float));
  
  level3::mat_ops_view view_x {
    .row_view = seq_len,
    .col_view = in_features,
    .leading_dimension = x.shape.strides[0],
    .data_view = x.tensor_data
  };
  
  level3::mat_ops_view view_w {
    .row_view = weight.shape.dims[0],
    .col_view = weight.shape.dims[1],
    .leading_dimension = weight.shape.strides[0],
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
    m->cfg.embed_dim = m->wte.shape.dims[1];
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
  
  std::printf("Detected config: vocab=%zu, embed=%zu, layers=%zu, heads=%zu, max_seq=%zu\n",
              m->cfg.vocab_size, m->cfg.embed_dim, m->cfg.num_layers,
              m->cfg.num_heads, m->cfg.max_seq_len);
  
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
    tens::tensor proj_weight = load_tensor(&sf, name, alloc);
    
    std::snprintf(name, sizeof(name), "h.%zu.attn.c_proj.bias", i);
    tens::tensor proj_bias = load_tensor(&sf, name, alloc);
   
    if (i == 0) {
      std::printf("[DEBUG] c_attn.weight shape: [%zu, %zu]\n", 
                qkv_weight.shape.dims[0], qkv_weight.shape.dims[1]);
    
      std::printf("[DEBUG] c_attn.weight[0][0]: %.6f\n", qkv_weight.tensor_data[0]);
      std::printf("[DEBUG] c_attn.weight[0][768]: %.6f\n", qkv_weight.tensor_data[768]);
      std::printf("[DEBUG] c_attn.weight[0][1536]: %.6f\n", qkv_weight.tensor_data[1536]);
    }


    float* w_q_buf = alloc.nn_alloc<float>(embed_dim * embed_dim);
    float* w_k_buf = alloc.nn_alloc<float>(embed_dim * embed_dim);
    float* w_v_buf = alloc.nn_alloc<float>(embed_dim * embed_dim);
    
    for (size_t row = 0; row < embed_dim; row++) {
      for (size_t col = 0; col < embed_dim; col++) {
        size_t src_stride = 3 * embed_dim;
        w_q_buf[row * embed_dim + col] = qkv_weight.tensor_data[row * src_stride + col];
        w_k_buf[row * embed_dim + col] = qkv_weight.tensor_data[row * src_stride + embed_dim + col];
        w_v_buf[row * embed_dim + col] = qkv_weight.tensor_data[row * src_stride + 2 * embed_dim + col];
      }
    }
    
    float* b_q = qkv_bias.tensor_data;
    float* b_k = qkv_bias.tensor_data + embed_dim;
    float* b_v = qkv_bias.tensor_data + 2 * embed_dim;
    float* w_o = proj_weight.tensor_data;
    float* b_o = proj_bias.tensor_data;
    
    void* pool_mem = alloc.nn_alloc<char>(sizeof(atten::atten_pool));
    void* atten_mem = alloc.nn_alloc<char>(sizeof(atten::multi_head_attention));
    
    m->atten_pools[i] = new (pool_mem) atten::atten_pool(atten_arena_size);
    m->attentions[i]  = new (atten_mem) atten::multi_head_attention(embed_dim, m->cfg.num_heads);
    m->attentions[i]->init(*m->atten_pools[i]);
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

    std::printf("[DEBUG] layer %zu ffn_fc_weight: [%zu, %zu]\n", 
            i, b->ffn_fc_weight.shape.dims[0], b->ffn_fc_weight.shape.dims[1]);
    std::printf("[DEBUG] layer %zu ffn_proj_weight: [%zu, %zu]\n", 
            i, b->ffn_proj_weight.shape.dims[0], b->ffn_proj_weight.shape.dims[1]);
  }
  
  m->ln_f_weight = load_tensor(&sf, "ln_f.weight", alloc);
  m->ln_f_bias = load_tensor(&sf, "ln_f.bias", alloc);
  
  safetensor::free_safetensor(&sf);
  
  m->initialized = true;
  std::printf("Model loaded successfully\n");
  return true;
}

tens::tensor forward(model *m, const tens::tensor &tokens, tens::tensor_pool &pool) {
    static int call_count = 0;
    bool debug = (call_count == 0);
    call_count++;
    
    size_t seq_len = tokens.shape.dims[0];
    size_t embed_dim = m->cfg.embed_dim;
    
    tens::tensor x = tens::ops::embedding(m->wte, tokens, pool);
    if (debug) debug_tensor("embedding", x);
    
    tens::tensor pos_emb;
    pos_emb.shape.ndim = 2;
    pos_emb.shape.dims[0] = seq_len;
    pos_emb.shape.dims[1] = embed_dim;
    pos_emb.shape.strides[0] = embed_dim;
    pos_emb.shape.strides[1] = 1;
    pos_emb.tensor_data = m->wpe.tensor_data;
    
    x = tens::ops::add(x, pos_emb, pool);
    if (debug) debug_tensor("after_pos", x);
    
for (size_t i = 0; i < m->cfg.num_layers; i++) {
    transformer_block* b = &m->blocks[i];
    
    tens::tensor residual = x;
    
    x = tens::ops::layer_norm(x, b->ln1_weight, b->ln1_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
    if (debug && i == 1) debug_tensor("layer1_ln1", x);
    
    x = m->attentions[i]->forward(x, *m->atten_pools[i]);
    if (debug && i == 1) debug_tensor("layer1_attn", x);
    
    x = tens::ops::add(residual, x, pool);
    if (debug && i == 1) debug_tensor("layer1_res1", x);
    
    residual = x;
    
    x = tens::ops::layer_norm(x, b->ln2_weight, b->ln2_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
    if (debug && i == 1) debug_tensor("layer1_ln2", x);
    
    x = linear(x, b->ffn_fc_weight, b->ffn_fc_bias, pool);
    if (debug && i == 1) debug_tensor("layer1_ffn_fc", x);
    
    x = tens::ops::gelu(x, pool);
    if (debug && i == 1) debug_tensor("layer1_gelu", x);
    
    x = linear(x, b->ffn_proj_weight, b->ffn_proj_bias, pool);
    if (debug && i == 1) debug_tensor("layer1_ffn_proj", x);
    
    x = tens::ops::add(residual, x, pool);
    
    if (debug) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "layer_%zu", i);
        debug_tensor(buf, x);
    }
}
    
    x = tens::ops::layer_norm(x, m->ln_f_weight, m->ln_f_bias, pool, x.shape.ndim - 1, m->cfg.layer_norm_eps);
    if (debug) debug_tensor("final_ln", x);
    
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
    
    level3::mat_ops_view view_wte {
        .row_view = m->cfg.vocab_size,
        .col_view = embed_dim,
        .leading_dimension = embed_dim,
        .data_view = m->wte.tensor_data
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
        view_wte,
        1.0f,
        0.0f,
        view_logits
    );
    
    if (debug) {
        debug_tensor("logits", logits);
        
        std::printf("Top 5 logits: ");
        float* l = logits.tensor_data;
        for (int k = 0; k < 5; k++) {
            int max_idx = 0;
            float max_val = -1e30f;
            for (size_t i = 0; i < m->cfg.vocab_size; i++) {
                if (l[i] > max_val) {
                    max_val = l[i];
                    max_idx = i;
                }
            }
            std::printf("%d(%.2f) ", max_idx, max_val);
            l[max_idx] = -1e30f;  // mask out for next iteration
        }
        std::printf("\n");
    }
    
    return logits;
}

int argmax(const tens::tensor& logits) {
  size_t vocab_size = logits.shape.dims[1];
  size_t seq_len = logits.shape.dims[0];
  
  float* last_row = logits.tensor_data + (seq_len - 1) * vocab_size;
  
  int max_idx = 0;
  float max_val = last_row[0];
  for (size_t i = 1; i < vocab_size; i++) {
    if (last_row[i] > max_val) {
      max_val = last_row[i];
      max_idx = i;
    }
  }
  return max_idx;
}

void free_model(model* m) {
  for (size_t i = 0; i < m->cfg.num_layers; i++) {
    if (m->attentions[i]) {
      m->attentions[i]->~multi_head_attention();
      m->attentions[i] = nullptr;
    }
    if (m->atten_pools[i]) {
      m->atten_pools[i]->~atten_pool();
      m->atten_pools[i] = nullptr;
    }
  }
  m->initialized = false;
}
}
