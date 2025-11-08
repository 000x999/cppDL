#include "../../include/tokenizer_core/tokenizer.hpp"

size_t bpe::token_pair_hash::operator()(const bpe::g_token_pair &pair) const{
  return std::hash<g_token_id>()(pair.first) ^ (std::hash<g_token_id>()(pair.second) << 1); 
}

bpe::bpe_tokenizer::bpe_tokenizer(){
  for(size_t i = 0; i < 256; ++i){
    std::string s(1, static_cast<char>(i));
    add_token(s); 
  }
}

void bpe::bpe_tokenizer::add_special_token(const std::vector<std::string> &special_token){
  for(const auto &token : special_token){
    add_token(token); 
  }
}

void bpe::bpe_tokenizer::train(const std::string &text, size_t merges, double regularization){
  if(text.empty() || merges == 0){
    return; 
  }
  std::vector<g_token_id> tokens; 
  for(auto &c : text){
    tokens.push_back(token_to_id[std::string(1,c)]); 
  }
  
  for(size_t merge = 0; merge < merges; ++merge){
    std::unordered_map<g_token_pair, int, token_pair_hash> token_pair_count; 
    for(size_t i = 0; i < tokens.size() - 1; ++i){
      g_token_pair pair = std::make_pair(tokens[i], tokens[i + 1]); 
      token_pair_count[pair]++; 
    }
    if(token_pair_count.empty()){
      break; 
    }
    using pair_count = std::pair<double, g_token_pair>; 
    std::priority_queue<pair_count> token_queue; 
    for(const auto &[pair, count] : token_pair_count){
      double score = count; 
      if(regularization > 0.0){
        const std::string &first  = id_to_token[pair.first]; 
        const std::string &second = id_to_token[pair.second];
        score -= regularization * (first.length() + second.length()); 
      }
      token_queue.emplace(score, pair); 
    }
    if(token_queue.empty()){
      break;
    }

    auto [max_token_score, best_token_pair] = token_queue.top(); 
    auto [a, b] = best_token_pair; 
    std::string merged = id_to_token[a] + id_to_token[b]; 
    g_token_id new_token_id = add_token(merged); 
    token_merge_history.push_back({a, b, new_token_id});

    std::vector<g_token_id> new_token; 
    size_t i =0; 
    while(i < tokens.size()){
      if(i < tokens.size() - 1 && tokens[i] == a && tokens[i + 1] == b){
        new_token.push_back(new_token_id);
        i += 2;
      }else{
        new_token.push_back(tokens[i]); 
        i++; 
      }
    }
    tokens = new_token; 
    std::cout << "create merge: " << merged << " (ID: " << new_token_id << " )\n"; 
  }
  std::cout << "total merges learned: " << token_merge_history.size() << std::endl; 
}
