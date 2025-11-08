#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <fstream> 
#include <sstream>
#include <utility>
#include <functional>
#include <memory>
#include <algorithm>
#include <stdexcept>

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

  std::vector<g_token_id> encode(const std::string &text) const {
    if (text.empty()) return {};

    std::vector<g_token_id> tokens;
    for (auto &c : text) {
      std::string s(1, c);
      try {
        tokens.push_back(token_to_id.at(s));
      } 
      catch (const std::out_of_range&) {
        tokens.push_back(UNK_TOKEN);
      }
    }

    for (const auto &rule : token_merge_history) {
      std::vector<g_token_id> new_token;
      size_t i = 0;
          
      while (i < tokens.size()) {
        if (i < tokens.size() - 1 && tokens[i] == rule.first && tokens[i+1] == rule.second) {
          new_token.push_back(rule.merged);
          i += 2;
        } else {
          new_token.push_back(tokens[i]);
          i++;
        }
      }
          
      tokens = std::move(new_token);
    }
      
    return tokens;
  }

  std::string decode(const std::vector<g_token_id> &tokens) const {
    std::string result;
    for (auto &id : tokens) {
      try {
        result += id_to_token.at(id);
      } 
      catch (const std::out_of_range&) {
        result += "<UNK>";
      }
    }
    return result;
  }

  bool save_model(const std::string &token_vocabulary_path, const std::string &token_merge_path) const {
    try {
      std::ofstream token_vocabulary_file(token_vocabulary_path);
      if (!token_vocabulary_file) return false;
      
      for (const auto &[token, id] : token_to_id) {
        token_vocabulary_file << token << "\t" << id << "\n";
      }
      token_vocabulary_file.close();
      
      std::ofstream token_merge_file(token_merge_path);
      if (!token_merge_file) return false;
      
      for (const auto &rule : token_merge_history) {
        token_merge_file << rule.first << "\t" << rule.second << "\t" << rule.merged << "\n";
      }
      token_merge_file.close();
      
      return true;
      } catch (const std::exception&) {
        return false;
      }
  }

  bool load_model(const std::string &token_vocabulary_path, const std::string &token_merge_path) {
    try {
      token_to_id.clear();
      id_to_token.clear();
      token_merge_history.clear();
      
      std::ifstream token_vocabulary_file(token_vocabulary_path);
      if (!token_vocabulary_file) return false;
      
      std::string token;
      g_token_id id;
      while (token_vocabulary_file >> token >> id) {
        token_to_id[token] = id;
        id_to_token[id] = token;
        if (id >= m_next_id) m_next_id = id + 1;
      }
      token_vocabulary_file.close();
      
      std::ifstream token_merge_file(token_merge_path);
      if (!token_merge_file) return false;
      
      g_token_id first, second, merged;
      while (token_merge_file >> first >> second >> merged) {
        token_merge_history.push_back({first, second, merged});
      }
      token_merge_file.close();
      
      return true;
      } catch (const std::exception&) {
        return false;
      }
  }

  void print_model_stats() const {
    std::cout << "Vocabulary size: " << token_to_id.size() << std::endl;
    std::cout << "Number of merge rules: " << token_merge_history.size() << std::endl;
      
    std::unordered_map<size_t, int> token_length_distance;
    for (const auto &[token, _] : token_to_id) {
      token_length_distance[token.length()]++;
    }
      
    std::cout << "Token length distribution:" << std::endl;
    for (const auto &[length, count] : token_length_distance) {
      std::cout << "  Length " << length << ": " << count << " tokens" << std::endl;
    }
  }

  size_t vocabulary_size() const {
    return token_to_id.size();
  }
  
  std::vector<std::string> get_vocabulary() const {
    std::vector<std::string> vocab(token_to_id.size());
    for (const auto& [token, id] : token_to_id) {
      if (id < vocab.size()) {
        vocab[id] = token;
      }
    }
    return vocab;
  }

  void debug_token(const std::string& token) const {
    try {
      std::cout << "Token '" << token << "' has ID: " << token_to_id.at(token) << std::endl;
    } 
    catch (const std::out_of_range&) {
      std::cout << "Token '" << token << "' not in vocabulary" << std::endl;
    }
  }

private:
  g_token_id add_token(const std::string &token) {
    if (token_to_id.find(token) == token_to_id.end()) {
      token_to_id[token] = m_next_id;
      id_to_token[m_next_id] = token;
      return m_next_id++;
    }
    return token_to_id[token];
  }

  g_token_id get_token_id(const std::string &token) const {
    try {
      return token_to_id.at(token);
    } 
    catch (const std::out_of_range&) {
      return UNK_TOKEN;
    }
  }
};


};
#endif
