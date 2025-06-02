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
using g_tokenid = uint32_t;
using g_tokenpair = std::pair<g_tokenid, g_tokenid>;

struct token_pair_hash {
  size_t operator()(const g_tokenpair &pair) const {
    return std::hash<g_tokenid>()(pair.first) ^ 
    (std::hash<g_tokenid>()(pair.second) << 1);
  }
};

class bpe_tokenizer {
private:
  std::unordered_map<std::string, g_tokenid> token_to_id;
  std::unordered_map<g_tokenid, std::string> id_to_token;
  g_tokenid m_nextId = 0;
  const g_tokenid UNK_TOKEN = UINT32_MAX;

  struct token_merge_rule {
    g_tokenid first;
    g_tokenid second;
    g_tokenid merged;
  };

public:
  std::vector<token_merge_rule> token_merge_history;
  
  bpe_tokenizer() {
    for (int i = 0; i < 256; ++i) {
      std::string s(1, static_cast<char>(i));
      add_token(s);
    }
  }

  void add_special_token(const std::vector<std::string>& special_token) {
    for (const auto& token : special_token) {
      add_token(token);
    }
  }

  void train(const std::string &text, size_t merges, double regularization = 0.0) {
    if (text.empty() || merges == 0) return;

    std::vector<g_tokenid> tokens;
    for (auto &c : text) {
      tokens.push_back(token_to_id[std::string(1, c)]);
    }

    for (size_t merge = 0; merge < merges; ++merge) {
      std::unordered_map<g_tokenpair, int, token_pair_hash> token_pair_count;
      
      for (size_t i = 0; i < tokens.size() - 1; ++i) {
        g_tokenpair pair = std::make_pair(tokens[i], tokens[i+1]);
        token_pair_count[pair]++;
      }

      if (token_pair_count.empty()) break;

      using pair_count = std::pair<double, g_tokenpair>;
      std::priority_queue<pair_count> token_queue;
      
      for (const auto &[pair, count] : token_pair_count) {
        double score = count;
        if (regularization > 0.0) {
          const std::string& first = id_to_token[pair.first];
          const std::string& second = id_to_token[pair.second];
          score -= regularization * (first.length() + second.length());
        }
        token_queue.emplace(score, pair);
      }

      if (token_queue.empty()) break;
      
      auto [max_token_score, best_token_pair] = token_queue.top();
      auto [a, b] = best_token_pair;
      
      std::string merged = id_to_token[a] + id_to_token[b];
      g_tokenid new_token_id = add_token(merged);
      
      token_merge_history.push_back({a, b, new_token_id});

      std::vector<g_tokenid> new_token;
      size_t i = 0;
      while (i < tokens.size()) {
        if (i < tokens.size() - 1 && tokens[i] == a && tokens[i+1] == b) {
          new_token.push_back(new_token_id);
          i += 2;
        } else {
          new_token.push_back(tokens[i]);
          i++;
        }
      }
      tokens = new_token;

      std::cout << "Created merge: " << merged << " (ID: " << new_token_id << ")\n";
    }
      
    std::cout << "Total merges learned: " << token_merge_history.size() << std::endl;
  }

  std::vector<g_tokenid> encode(const std::string &text) const {
    if (text.empty()) return {};

    std::vector<g_tokenid> tokens;
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
      std::vector<g_tokenid> new_token;
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

  std::string decode(const std::vector<g_tokenid> &tokens) const {
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
      g_tokenid id;
      while (token_vocabulary_file >> token >> id) {
        token_to_id[token] = id;
        id_to_token[id] = token;
        if (id >= m_nextId) m_nextId = id + 1;
      }
      token_vocabulary_file.close();
      
      std::ifstream token_merge_file(token_merge_path);
      if (!token_merge_file) return false;
      
      g_tokenid first, second, merged;
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
  g_tokenid add_token(const std::string &token) {
    if (token_to_id.find(token) == token_to_id.end()) {
      token_to_id[token] = m_nextId;
      id_to_token[m_nextId] = token;
      return m_nextId++;
    }
    return token_to_id[token];
  }

  g_tokenid get_token_id(const std::string &token) const {
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






























