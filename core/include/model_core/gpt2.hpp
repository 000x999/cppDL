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
    std::unordered_map<std::string, int> token_to_id; 

    int hex_val(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    }

    void append_utf8(std::string& out, int cp) {
        if (cp <= 0x7F) out += (char)cp;
        else if (cp <= 0x7FF) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
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

            std::string raw = content.substr(key_start + 1, key_end - key_start - 1);
            int id = std::stoi(content.substr(val_start, val_end - val_start));
            
            std::string decoded = "";
            for (size_t i = 0; i < raw.length(); i++) {
                if (raw[i] == '\\' && i + 5 < raw.length() && raw[i+1] == 'u') {
                    int cp = (hex_val(raw[i+2]) << 12) | (hex_val(raw[i+3]) << 8) |
                             (hex_val(raw[i+4]) << 4)  | hex_val(raw[i+5]);
                    append_utf8(decoded, cp);
                    i += 5;
                } else if (raw[i] == '\\' && i + 1 < raw.length()) {
                     decoded += raw[i+1]; 
                     i++;
                } else {
                    decoded += raw[i];
                }
            }
            id_to_token[id] = decoded;
            token_to_id[decoded] = id; 
            pos = val_end + 1;
        }
        return true;
    }

    std::string decode(int id) {
        if (id_to_token.find(id) == id_to_token.end()) return "";
        std::string raw = id_to_token[id];
        std::string out = "";
        
        for (size_t i = 0; i < raw.length(); ) {
            if (i+1 < raw.length() && (unsigned char)raw[i]==0xC4 && (unsigned char)raw[i+1]==0xA0) {
                out += " "; i += 2;
            } else if (i+1 < raw.length() && (unsigned char)raw[i]==0xC4 && (unsigned char)raw[i+1]==0x8A) {
                out += "\n"; i += 2;
            } else {
                out += raw[i]; i++;
            }
        }
        return out;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        
        std::string current_chunk;
        bool is_start_of_word = true; 

        auto commit_chunk = [&](std::string chunk, bool prepend_space) {
            if (chunk.empty()) return;
            
            std::string search = chunk;
            if (prepend_space) {
                std::string tmp = ""; 
                tmp += (char)0xC4; tmp += (char)0xA0; 
                tmp += chunk;
                
                if (token_to_id.count(tmp)) search = tmp;
            }

            if (token_to_id.count(search)) {
                tokens.push_back(token_to_id[search]);
            } else if (token_to_id.count(chunk)) {
                tokens.push_back(token_to_id[chunk]);
            } else {
                std::printf("[WARN] Unknown token: '%s'\n", chunk.c_str());
            }
        };

        for (size_t i = 0; i < text.length(); i++) {
            char c = text[i];
            
            if (std::isspace(c)) {
                commit_chunk(current_chunk, !is_start_of_word);
                current_chunk = "";
                is_start_of_word = false;
            } 
            else if (std::ispunct(c) && c != '\'') {
                commit_chunk(current_chunk, !is_start_of_word);
                current_chunk = "";
                
                std::string p_str(1, c);
                commit_chunk(p_str, false); 
                
                is_start_of_word = true; 
            } 
            else {
                current_chunk += c;
            }
        }
        commit_chunk(current_chunk, !is_start_of_word);
        
      return tokens;
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
  memory::neural_arena        * atten_pools[MAX_LAYERS];
  tens::tensor ln_f_weight;
  tens::tensor ln_f_bias;
  bool initialized;
};

void         init_model          (model* m                                                     );
bool         load_model          (model* m,const char* path, memory::neural_arena& alloc       );
tens::tensor forward             (model* m,const tens::tensor& tokens, memory::neural_arena& pool  );
int          argmax              (const tens::tensor& logits                                   );
void         free_model          (model* m                                                     );
int          sample_top_k_avx512 (float* logits, size_t vocab_size, int k, float temperature, memory::neural_arena& pool);
int          sample_top_k        (float* logits, size_t vocab_size, int k);
tens::tensor matmul              (const tens::tensor &a, const tens::tensor &b, memory::neural_arena &pool);
}  // namespace gpt2
