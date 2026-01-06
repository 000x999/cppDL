#pragma once
#include "safetensor_core/safetensor_reader.h"
#include "tensor_core/tensor.hpp"
#include "neural_memory/nn_memory.hpp"
#include "attention_core/attention.hpp"

namespace gpt2 {
constexpr size_t MAX_LAYERS  = 48;
constexpr size_t MAX_SEQ_LEN = 2048;

struct config {
  size_t vocab_size;
  size_t embed_dim;
  size_t num_heads;
  size_t num_layers;
  size_t max_seq_len;
  float layer_norm_eps;
};

struct transformer_block {
  tens::tensor ln1_weight;
  tens::tensor ln1_bias;
  tens::tensor ln2_weight;
  tens::tensor ln2_bias;
  tens::tensor ffn_fc_weight;
  tens::tensor ffn_fc_bias;
  tens::tensor ffn_proj_weight;
  tens::tensor ffn_proj_bias;
};

struct model {
  config cfg;
  tens::tensor wte;
  tens::tensor wpe;
  tens::tensor wte_T;
  transformer_block blocks[MAX_LAYERS];
  atten::multi_head_attention * attentions [MAX_LAYERS];
  atten::atten_pool           * atten_pools[MAX_LAYERS];
  tens::tensor ln_f_weight;
  tens::tensor ln_f_bias;
  bool initialized;
};

void         init_model (model* m                                                    );
bool         load_model (model* m,const char* path,memory::neural_arena& alloc       );
tens::tensor forward    (model* m,const tens::tensor& tokens,tens::tensor_pool& pool );
int          argmax     (const tens::tensor& logits                                  );
void         free_model (model* m                                                    );
tens::tensor matmul     (const tens::tensor &a, const tens::tensor &b, tens::tensor_pool &pool);
}  // namespace gpt2
