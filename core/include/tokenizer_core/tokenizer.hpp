#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <fstream> 
#include <sstream>
#include <numeric> 
#include <limits>
#include <thread>
#include <chrono>
#include <cmath>
#include <utility>
#include <functional>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>

namespace bpe{
using g_token_id = uint32_t;
using g_token_pair = std::pair<g_token_id, g_token_id>;

struct token_pair_hash {
  size_t operator()(const g_token_pair &pair) const;
};

class bpe_tokenizer {
private:
  std::unordered_map<std::string, g_token_id> token_to_id;
  std::unordered_map<g_token_id, std::string> id_to_token;
  std::vector<g_token_id> token_length; 
  g_token_id m_next_id = 0;
  const g_token_id UNK_TOKEN = UINT32_MAX;

  struct token_merge_rule {
    g_token_id first;
    g_token_id second;
    g_token_id merged;
  };

public:
  std::vector<token_merge_rule> token_merge_history;
  
  bpe_tokenizer(); 
  void add_special_token(const std::vector<std::string>& special_token);
  void train(const std::string &text, size_t merges, double regularization = 0.0);

  std::vector<g_token_id> encode(const std::string &text) const; 
  std::string decode(const std::vector<g_token_id> &tokens) const; 
  bool save_model(const std::string &token_vocabulary_path, const std::string &token_merge_path) const;
  bool load_model(const std::string &token_vocabulary_path, const std::string &token_merge_path); 
  void print_model_stats() const; 
  size_t vocabulary_size() const;
  std::vector<std::string> get_vocabulary() const;
  void debug_token(const std::string& token) const; 

private:
  g_token_id add_token(const std::string &token);
  g_token_id get_token_id(const std::string &token) const;
};


};
#endif
