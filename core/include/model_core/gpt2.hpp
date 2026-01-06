#pragma once
#include "safetensor_core/safetensor_reader.h"
#include "tensor_core/tensor.hpp"
#include "neural_memory/nn_memory.hpp"
#include "attention_core/attention.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace gpt2 {
constexpr size_t MAX_LAYERS  = 48;
constexpr size_t MAX_SEQ_LEN = 2048;

struct Token {
  float score;
  int id;
};

struct tokenizer {
    std::unordered_map<int, std::string> id_to_token;
    
    std::string clean_token(const std::string& token) {
        std::string out = "";
        for (size_t i = 0; i < token.length(); ) {
            if (i + 1 < token.length() && 
                (unsigned char)token[i] == 0xC4 && 
                (unsigned char)token[i+1] == 0xA0) {
                out += " ";
                i += 2;
            } else {
                out += token[i];
                i++;
            }
        }
        return out;
    }

    bool load(const char* vocab_path) {
        std::ifstream f(vocab_path);
        if (!f.is_open()) return false;

        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string content = buffer.str();

        size_t pos = 0;
        while (true) {
            size_t key_start = content.find('"', pos);
            if (key_start == std::string::npos) break;
            
            size_t key_end = content.find('"', key_start + 1);
            if (key_end == std::string::npos) break;

            size_t val_start = content.find_first_of("0123456789", key_end + 1);
            if (val_start == std::string::npos) break;

            size_t val_end = content.find_first_not_of("0123456789", val_start);
            if (val_end == std::string::npos) val_end = content.length();

            std::string token_raw = content.substr(key_start + 1, key_end - key_start - 1);
            std::string id_str = content.substr(val_start, val_end - val_start);
            
            try {
                int id = std::stoi(id_str);
                
                std::string token_decoded = "";
                for (size_t i = 0; i < token_raw.length(); i++) {
                    if (token_raw[i] == '\\' && i + 1 < token_raw.length()) {
                        if (token_raw[i+1] == 'u') {
                             i += 5;
                        } else {
                            token_decoded += token_raw[i+1];
                            i++;
                        }
                    } else {
                        token_decoded += token_raw[i];
                    }
                }
                
                if (token_decoded.empty()) token_decoded = token_raw;
                
                id_to_token[id] = token_decoded;
            } catch (...) {}

            pos = val_end + 1;
        }
        
        std::printf("Loaded tokenizer vocab: %zu tokens\n", id_to_token.size());
        return true;
    }

    std::string decode(int id) {
        if (id_to_token.find(id) != id_to_token.end()) {
            return clean_token(id_to_token[id]);
        }
        return ""; 
    }
};

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

void         init_model          (model* m                                                     );
bool         load_model          (model* m,const char* path, memory::neural_arena& alloc       );
tens::tensor forward             (model* m,const tens::tensor& tokens,tens::tensor_pool& pool  );
int          argmax              (const tens::tensor& logits                                   );
void         free_model          (model* m                                                     );
int          sample_top_k_avx512 (float* logits, size_t vocab_size, int k, float temperature, tens::tensor_pool& pool);
int          sample_top_k        (float* logits, size_t vocab_size, int k);
tens::tensor matmul              (const tens::tensor &a, const tens::tensor &b, tens::tensor_pool &pool);
}  // namespace gpt2
