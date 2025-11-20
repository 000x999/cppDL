#include "../../include/tokenizer_core/tokenizer.hpp"

namespace{
struct token_node{
  bpe::g_token_id id;
  int previous_token; 
  int next_token; 
  bool is_alive; 
};

struct token_pair_key{
  bpe::g_token_id id_a; 
  bpe::g_token_id id_b; 

  bool operator==(const token_pair_key &other_pair) const noexcept{
    return id_a == other_pair.id_a && id_b == other_pair.id_b; 
  }
};

struct token_pair_key_hash{
  size_t operator()(const token_pair_key &token_key) const noexcept{
    return (static_cast<size_t>(token_key.id_a) << 32) | static_cast<size_t>(token_key.id_b); 
  }
};

struct token_pair_info{
  int token_count   = 0;
  double pair_score = 0.0; 
  bool is_active    = true; 
  std::vector<int> token_positions; 
};

struct token_heap_entry{
  double heap_score = 0.0; 
  token_pair_key token_key; 
};

struct token_heap_compare{
  bool operator()(const token_heap_entry &left_heap, const token_heap_entry &right_heap) const noexcept{
    return left_heap.heap_score < right_heap.heap_score; 
  }
};

}//namespace 

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

  token_merge_history.clear();

  std::vector<token_node> model_nodes;
  model_nodes.reserve(text.size()); 
  
  for(size_t i = 0; i < text.size(); ++i){
    std::string s(1, text[i]); 
    g_token_id token_id = get_token_id(s);

    token_node node; 
    node.id             = token_id;
    node.previous_token = (i == 0) ? -1 : static_cast<int>(i - 1);
    node.next_token     = (i + 1 < text.size()) ? static_cast<int>(i + 1) : - 1; 
    node.is_alive       = true;
    model_nodes.push_back(node);
  }

  int head_node = model_nodes.empty() ? -1 : 0;

  using token_pair_map = std::unordered_map<token_pair_key, token_pair_info, token_pair_key_hash>;
  token_pair_map pair_map; 
  pair_map.reserve(text.size() * 2);

  auto make_token_key = [](g_token_id token_a, g_token_id token_b) -> token_pair_key{
    return token_pair_key{token_a, token_b}; 
  };

  int current_node = head_node;
  while(current_node != -1){
    int next_node = model_nodes[current_node].next_token;
    if(next_node != -1){
      token_pair_key  pair_key   = make_token_key(model_nodes[current_node].id, model_nodes[next_node].id); 
      token_pair_info &pair_info = pair_map[pair_key];
      pair_info.token_count++; 
      pair_info.token_positions.push_back(current_node);
    }
    current_node = next_node; 
  }

  std::priority_queue<token_heap_entry, std::vector<token_heap_entry>, token_heap_compare> model_heap; 
  auto compute_pair_score = [&](const token_pair_key &token_key, const token_pair_info &pair_info) -> double{
    double score = static_cast<double>(pair_info.token_count); 
    if(regularization > 0.0){
        score -= regularization * (token_length[token_key.id_a] + token_length[token_key.id_b]); 
    }
    return score; 
  };

  for(auto &map_entry : pair_map){
    const token_pair_key &token_key = map_entry.first; 
    token_pair_info &pair_info      = map_entry.second; 
    if(pair_info.token_count <= 0){continue;}
    pair_info.pair_score = compute_pair_score(token_key, pair_info);
    model_heap.push(token_heap_entry{pair_info.pair_score, token_key}); 
  }

  auto increment_token_pair = [&](const token_pair_key &token_key, int left_token_position){
    token_pair_info &pair_info = pair_map[token_key]; 
    pair_info.token_count++; 
    pair_info.token_positions.push_back(left_token_position);
    if(pair_info.token_count >= 2){
      pair_info.pair_score = compute_pair_score(token_key, pair_info); 
      model_heap.push(token_heap_entry{pair_info.pair_score, token_key}); 
    }
  }; 

  auto decrement_token_pair = [&](const token_pair_key &token_key){
    auto it = pair_map.find(token_key); 
    if(it == pair_map.end()){return;}
    token_pair_info &pair_info = it->second; 
    if(pair_info.token_count <= 0){return;}
    pair_info.token_count--;
    if(pair_info.token_count >= 2){
      pair_info.pair_score = compute_pair_score(token_key, pair_info);
      model_heap.push(token_heap_entry{pair_info.pair_score, token_key}); 
    }
  };
  
  size_t merges_done = 0; 
  while(merges_done < merges && !model_heap.empty()){
    token_pair_key   best_pair_key{}; 
    token_pair_info *best_pair_info = nullptr; 

    while(!model_heap.empty()){
      token_heap_entry top_entry = model_heap.top();
      model_heap.pop();

      auto it = pair_map.find(top_entry.token_key);
      if(it == pair_map.end()){
        continue; 
      }
      token_pair_info &pair_info = it->second;
      if(!pair_info.is_active || pair_info.token_count <= 0){
        continue; 
      }

      double current_score = pair_info.pair_score; 
      if(std::fabs(current_score - top_entry.heap_score) > 1e-9){
        continue; 
      }
      best_pair_key  = top_entry.token_key; 
      best_pair_info = &pair_info;
      break;
    }
    if(!best_pair_info){break;}

    if(best_pair_info->token_count < 2){break;}

    g_token_id token_a = best_pair_key.id_a; 
    g_token_id token_b = best_pair_key.id_b;

    std::string merged_token_string = id_to_token[token_a] + id_to_token[token_b]; 
    g_token_id  merged_token_id     = add_token(merged_token_string); 
    
    token_merge_history.push_back({token_a, token_b, merged_token_id});

    std::cout << "[BPE] MERGES :: " << (merges_done + 1)
              << ": \"" << id_to_token[token_a] << "\" (" << token_a << ") + "
              << "\"" << id_to_token[token_b] << "\" (" << token_b << ") -> "
              << "\"" << merged_token_string << "\" (" << merged_token_id << ")\n";
    std::cout << "  COUNT  :: " << best_pair_info->token_count
              << "  SCORE  ::" << best_pair_info->pair_score * 1e-9 << "\n";
    std::cout << "    MERGED TOKEN :: \"" << merged_token_string
              << "\" (" << merged_token_id << ")\n";
    
    merges_done++; 
    best_pair_info->is_active = false;

    for(int pos : best_pair_info->token_positions){
      if(pos < 0 || pos >= static_cast<int>(model_nodes.size())){continue;}
      if(!model_nodes[pos].is_alive){continue;}

      int i = pos; 
      int j = model_nodes[i].next_token;
      if(j == -1){continue;}
      if(!model_nodes[j].is_alive){continue;}

      if(model_nodes[i].id != token_a || model_nodes[j].id != token_b){
        continue;
      }

      int left_node  = model_nodes[i].previous_token; 
      int right_node = model_nodes[j].next_token;
    
      if(left_node != -1 && model_nodes[left_node].is_alive){
        token_pair_key left_pair{model_nodes[left_node].id, token_a};
        decrement_token_pair(left_pair); 
      }
      
      if(right_node != -1 && model_nodes[right_node].is_alive){
        token_pair_key old_right_pair{token_b, model_nodes[right_node].id}; 
        decrement_token_pair(old_right_pair); 
      }
      model_nodes[i].id         = merged_token_id;
      model_nodes[i].next_token = right_node;
    
      if(right_node != -1){
        model_nodes[right_node].previous_token = i;
      }
      model_nodes[j].is_alive       = false; 
      model_nodes[j].previous_token = -1;
      model_nodes[j].next_token     = -1;

      if(left_node != -1 && model_nodes[left_node].is_alive){
        token_pair_key new_left_pair{model_nodes[left_node].id, merged_token_id};
        increment_token_pair(new_left_pair, left_node); 
      }
      
      if(right_node != -1 && model_nodes[right_node].is_alive){
        token_pair_key new_right_pair{merged_token_id, model_nodes[right_node].id};
        increment_token_pair(new_right_pair, i);
      }
    }
  }
  std::cout << "TOTAL LEARNED MERGES ::" << token_merge_history.size() << '\n'; 
}

std::vector<bpe::g_token_id> bpe::bpe_tokenizer::encode(const std::string &text) const{
  if(text.empty()) return {}; 
  std::vector<g_token_id> tokens; 

  for(auto &c : text){
    std::string s(1, c); 
    try{
      tokens.push_back(token_to_id.at(s)); 
    }catch(const std::out_of_range&){
      tokens.push_back(UNK_TOKEN); 
    }
  }
  for(const auto &rule : token_merge_history){
    std::vector<g_token_id> new_token; 
    size_t i = 0; 

    while(i < tokens.size()){
      if(i < tokens.size() - 1 && tokens[i] == rule.first && tokens[i + 1] == rule.second){
        new_token.push_back(rule.merged); 
        i += 2; 
      }else{
        new_token.push_back(tokens[i]); 
        ++i; 
      }
    }
    tokens = std::move(new_token); 
  }
  return tokens; 
}

std::string bpe::bpe_tokenizer::decode(const std::vector<g_token_id> &tokens) const{
  std::string result; 
  for(auto &id : tokens){
    try{
      result += id_to_token.at(id); 
    }catch(const std::out_of_range&){
      result += "<UNK>";
    }
  }
  return result;  
}

bool bpe::bpe_tokenizer::save_model(const std::string &token_vocabulary_path, const std::string &token_merge_path) const{
  try{
    std::ofstream token_vocabulary_file(token_vocabulary_path); 
    if(!token_vocabulary_file){return false;}
    
    for(const auto &[token, id] : token_to_id){
      token_vocabulary_file << token << "\t" << id << "\n";   
    }
    token_vocabulary_file.close(); 

    std::ofstream token_merge_file(token_merge_path);
    if(!token_merge_file){return false;}

    for(const auto &rule : token_merge_history){
      token_merge_file << rule.first << "\t" << rule.second << "\t" << rule.merged << "\n"; 
    }
    token_merge_file.close(); 

    return true; 

  }catch(const std::exception&){
    return false; 
  }
}

bool bpe::bpe_tokenizer::load_model(const std::string &token_vocabulary_path, const std::string &token_merge_path){
  try{
    token_to_id.clear(); 
    id_to_token.clear(); 
    token_merge_history.clear();
  
    std::ifstream token_vocabulary_file(token_vocabulary_path); 
    if(!token_vocabulary_file){return false;}
    
    std::string token;
    g_token_id id; 
    while(token_vocabulary_file >> token >> id){
      token_to_id[token] = id;
      id_to_token[id]    = token; 
      if(id >= m_next_id){m_next_id = id + 1;}
    }
    token_vocabulary_file.close();

    std::ifstream token_merge_file(token_merge_path); 
    if(!token_merge_file){return false;}

    g_token_id first; 
    g_token_id second; 
    g_token_id merged; 
    while(token_merge_file >> first >> second >> merged){
      token_merge_history.push_back({first, second, merged}); 
    }
    token_merge_file.close(); 

    return true;

  }catch(const std::exception&){
    return false; 
  }
}

void bpe::bpe_tokenizer::print_model_stats()const{
  std::cout << "MODEL VOCABULARY SIZE :: " << token_to_id.size() << '\n';
  std::cout << "NUMBER OF MERGE RULES :: " << token_merge_history.size() << '\n';

  std::unordered_map<size_t, int> token_length_distance; 
  for(const auto &[token, _] : token_to_id){
    token_length_distance[token.length()]++;
  }
  std::cout << "TOKEN LENGTH DIST :: "; 
  for(const auto &[length, count] : token_length_distance){
    std::cout << "LENGTH :: " << length << ":: " << count << " TOKENS" << '\n';   
  }
}

size_t bpe::bpe_tokenizer::vocabulary_size() const{
  return token_to_id.size(); 
}

std::vector<std::string> bpe::bpe_tokenizer::get_vocabulary() const{
  std::vector<std::string> vocab(token_to_id.size());
  for(const auto &[token, id] : token_to_id){
    if(id < vocab.size()){
      vocab[id] = token; 
    }
  }
  return vocab; 
}

void bpe::bpe_tokenizer::debug_token(const std::string &token) const{
  try{
    std::cout << "TOKEN :: '" << token << "' HAS ID :: " << token_to_id.at(token) << '\n';  
  }catch(const std::out_of_range&){
    std::cout << "TOKEN :: '" << token << "' NOT IN THE VOCABULARY" << '\n'; 
  }
}

bpe::g_token_id bpe::bpe_tokenizer::add_token(const std::string &token){
  /*
  if(token_to_id.find(token) == token_to_id.end()){
    token_to_id[token] = m_next_id;
    id_to_token[m_next_id] = token;
    return m_next_id++; 
  }
  return token_to_id[token];
  */
  auto it = token_to_id.find(token); 
  if(it != token_to_id.end()){
    return it->second;
  }
  g_token_id id = m_next_id++; 
  token_to_id[token] = id; 
  id_to_token[id] = token; 

  if(id >= token_length.size()){
    token_length.resize(id + 1); 
  }
  token_length[id] = static_cast<g_token_id>(token.size());
  return id;
}

bpe::g_token_id bpe::bpe_tokenizer::get_token_id(const std::string &token) const{
  try{
    return token_to_id.at(token);
  }catch(const std::out_of_range&){
    return UNK_TOKEN; 
  }
}
