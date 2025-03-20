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

namespace BPE{
using g_tokenid = uint32_t;
using g_tokenpair = std::pair<g_tokenid, g_tokenid>;

struct TokenPairHash {
  size_t operator()(const g_tokenpair &pair) const {
    return std::hash<g_tokenid>()(pair.first) ^ 
    (std::hash<g_tokenid>()(pair.second) << 1);
  }
};

class BPETokenizer {
private:
  std::unordered_map<std::string, g_tokenid> tokenToId;
  std::unordered_map<g_tokenid, std::string> idToToken;
  g_tokenid m_nextId = 0;
  const g_tokenid UNK_TOKEN = UINT32_MAX;

  struct MergingRule {
    g_tokenid first;
    g_tokenid second;
    g_tokenid merged;
  };

public:
  std::vector<MergingRule> mergeHist;
  
  BPETokenizer() {
    for (int i = 0; i < 256; ++i) {
      std::string s(1, static_cast<char>(i));
      addToken(s);
    }
  }

  void addSpecialTokens(const std::vector<std::string>& specialTokens) {
    for (const auto& token : specialTokens) {
      addToken(token);
    }
  }

  void train(const std::string &text, size_t merges, double regularization = 0.0) {
    if (text.empty() || merges == 0) return;

    std::vector<g_tokenid> tokens;
    for (auto &c : text) {
      tokens.push_back(tokenToId[std::string(1, c)]);
    }

    for (size_t merge = 0; merge < merges; ++merge) {
      std::unordered_map<g_tokenpair, int, TokenPairHash> tokenPairCounts;
      
      for (size_t i = 0; i < tokens.size() - 1; ++i) {
        g_tokenpair pair = std::make_pair(tokens[i], tokens[i+1]);
        tokenPairCounts[pair]++;
      }

      if (tokenPairCounts.empty()) break;

      using PairCount = std::pair<double, g_tokenpair>;
      std::priority_queue<PairCount> tokenQueue;
      
      for (const auto &[pair, count] : tokenPairCounts) {
        double score = count;
        if (regularization > 0.0) {
          const std::string& first = idToToken[pair.first];
          const std::string& second = idToToken[pair.second];
          score -= regularization * (first.length() + second.length());
        }
        tokenQueue.emplace(score, pair);
      }

      if (tokenQueue.empty()) break;
      
      auto [maxScore, bestPair] = tokenQueue.top();
      auto [a, b] = bestPair;
      
      std::string merged = idToToken[a] + idToToken[b];
      g_tokenid newId = addToken(merged);
      
      mergeHist.push_back({a, b, newId});

      std::vector<g_tokenid> newTokens;
      size_t i = 0;
      while (i < tokens.size()) {
        if (i < tokens.size() - 1 && tokens[i] == a && tokens[i+1] == b) {
          newTokens.push_back(newId);
          i += 2;
        } else {
          newTokens.push_back(tokens[i]);
          i++;
        }
      }
      tokens = newTokens;

      std::cout << "Created merge: " << merged << " (ID: " << newId << ")\n";
    }
      
    std::cout << "Total merges learned: " << mergeHist.size() << std::endl;
  }

  std::vector<g_tokenid> encode(const std::string &text) const {
    if (text.empty()) return {};

    std::vector<g_tokenid> tokens;
    for (auto &c : text) {
      std::string s(1, c);
      try {
        tokens.push_back(tokenToId.at(s));
      } 
      catch (const std::out_of_range&) {
        tokens.push_back(UNK_TOKEN);
      }
    }

    for (const auto &rule : mergeHist) {
      std::vector<g_tokenid> newTokens;
      size_t i = 0;
          
      while (i < tokens.size()) {
        if (i < tokens.size() - 1 && tokens[i] == rule.first && tokens[i+1] == rule.second) {
          newTokens.push_back(rule.merged);
          i += 2;
        } else {
          newTokens.push_back(tokens[i]);
          i++;
        }
      }
          
      tokens = std::move(newTokens);
    }
      
    return tokens;
  }

  std::string decode(const std::vector<g_tokenid> &tokens) const {
    std::string result;
    for (auto &id : tokens) {
      try {
        result += idToToken.at(id);
      } 
      catch (const std::out_of_range&) {
        result += "<UNK>";
      }
    }
    return result;
  }

  bool saveModel(const std::string &vocabPath, const std::string &mergesPath) const {
    try {
      std::ofstream vocabFile(vocabPath);
      if (!vocabFile) return false;
      
      for (const auto &[token, id] : tokenToId) {
        vocabFile << token << "\t" << id << "\n";
      }
      vocabFile.close();
      
      std::ofstream mergesFile(mergesPath);
      if (!mergesFile) return false;
      
      for (const auto &rule : mergeHist) {
        mergesFile << rule.first << "\t" << rule.second << "\t" << rule.merged << "\n";
      }
      mergesFile.close();
      
      return true;
      } catch (const std::exception&) {
        return false;
      }
  }

  bool loadModel(const std::string &vocabPath, const std::string &mergesPath) {
    try {
      tokenToId.clear();
      idToToken.clear();
      mergeHist.clear();
      
      std::ifstream vocabFile(vocabPath);
      if (!vocabFile) return false;
      
      std::string token;
      g_tokenid id;
      while (vocabFile >> token >> id) {
        tokenToId[token] = id;
        idToToken[id] = token;
        if (id >= m_nextId) m_nextId = id + 1;
      }
      vocabFile.close();
      
      std::ifstream mergesFile(mergesPath);
      if (!mergesFile) return false;
      
      g_tokenid first, second, merged;
      while (mergesFile >> first >> second >> merged) {
        mergeHist.push_back({first, second, merged});
      }
      mergesFile.close();
      
      return true;
      } catch (const std::exception&) {
        return false;
      }
  }

  void printStats() const {
    std::cout << "Vocabulary size: " << tokenToId.size() << std::endl;
    std::cout << "Number of merge rules: " << mergeHist.size() << std::endl;
      
    std::unordered_map<size_t, int> lengthDist;
    for (const auto &[token, _] : tokenToId) {
      lengthDist[token.length()]++;
    }
      
    std::cout << "Token length distribution:" << std::endl;
    for (const auto &[length, count] : lengthDist) {
      std::cout << "  Length " << length << ": " << count << " tokens" << std::endl;
    }
  }

  size_t vocabSize() const {
    return tokenToId.size();
  }
  
  std::vector<std::string> getVocab() const {
    std::vector<std::string> vocab(tokenToId.size());
    for (const auto& [token, id] : tokenToId) {
      if (id < vocab.size()) {
        vocab[id] = token;
      }
    }
    return vocab;
  }

  void debugToken(const std::string& token) const {
    try {
      std::cout << "Token '" << token << "' has ID: " << tokenToId.at(token) << std::endl;
    } 
    catch (const std::out_of_range&) {
      std::cout << "Token '" << token << "' not in vocabulary" << std::endl;
    }
  }

private:
  g_tokenid addToken(const std::string &token) {
    if (tokenToId.find(token) == tokenToId.end()) {
      tokenToId[token] = m_nextId;
      idToToken[m_nextId] = token;
      return m_nextId++;
    }
    return tokenToId[token];
  }

  g_tokenid getId(const std::string &token) const {
    try {
      return tokenToId.at(token);
    } 
    catch (const std::out_of_range&) {
      return UNK_TOKEN;
    }
  }
};


};
#endif






























