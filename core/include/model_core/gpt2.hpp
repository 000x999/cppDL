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
#include <regex>
#include <algorithm>

namespace gpt2 {
constexpr size_t MAX_LAYERS  = 48;
constexpr size_t MAX_SEQ_LEN = 2048;

struct Token {
  float score;
  int id;
};

static inline void append_utf8(std::string& out, uint32_t cp) {
  if (cp <= 0x7F) {
    out.push_back((char)cp);
  } else if (cp <= 0x7FF) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

static inline bool next_codepoint_utf8(const std::string& s, size_t& i, uint32_t& cp) {
  if (i >= s.size()) return false;
  uint8_t c0 = (uint8_t)s[i++];

  if ((c0 & 0x80) == 0) { cp = c0; return true; }

  if ((c0 & 0xE0) == 0xC0) {
    if (i >= s.size()) return false;
    uint8_t c1 = (uint8_t)s[i++];
    cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    return true;
  }

  if ((c0 & 0xF0) == 0xE0) {
    if (i + 1 >= s.size()) return false;
    uint8_t c1 = (uint8_t)s[i++];
    uint8_t c2 = (uint8_t)s[i++];
    cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    return true;
  }

  if ((c0 & 0xF8) == 0xF0) {
    if (i + 2 >= s.size()) return false;
    uint8_t c1 = (uint8_t)s[i++];
    uint8_t c2 = (uint8_t)s[i++];
    uint8_t c3 = (uint8_t)s[i++];
    cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    return true;
  }

  return false;
}

static inline std::vector<std::string> split_utf8_codepoints(const std::string& s) {
  std::vector<std::string> cps;
  cps.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    size_t start = i;
    uint8_t c0 = (uint8_t)s[i++];
    if ((c0 & 0x80) == 0) {
      // 1-byte
    } else if ((c0 & 0xE0) == 0xC0) {
      i = std::min(i + 1, s.size());
    } else if ((c0 & 0xF0) == 0xE0) {
      i = std::min(i + 2, s.size());
    } else if ((c0 & 0xF8) == 0xF0) {
      i = std::min(i + 3, s.size());
    }
    cps.emplace_back(s.substr(start, i - start));
  }
  return cps;
}

static inline std::string join_with_space(const std::vector<std::string>& parts) {
  if (parts.empty()) return "";
  std::string out = parts[0];
  for (size_t i = 1; i < parts.size(); i++) {
    out.push_back(' ');
    out += parts[i];
  }
  return out;
}

struct tokenizer {
  std::unordered_map<int, std::string> id_to_token;   
  std::unordered_map<std::string, int> token_to_id;   

  std::unordered_map<std::string, int> bpe_ranks;
  std::unordered_map<std::string, std::string> bpe_cache;

  std::array<std::string, 256> byte_to_unicode_utf8;        
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte;    

  static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }

  static void append_utf8_cp(std::string& out, int cp) { append_utf8(out, (uint32_t)cp); }

  void init_gpt2_byte_maps() {
    unicode_to_byte.clear();

    bool used[256] = {false};
    std::vector<int> bs;
    std::vector<int> cs;

    auto push_range = [&](int a, int b_exclusive) {
      for (int x = a; x < b_exclusive; x++) {
        bs.push_back(x);
        used[x] = true;
      }
    };

    push_range(33, 127);
    push_range(161, 173);
    push_range(174, 256);

    cs = bs;

    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!used[b]) {
        bs.push_back(b);
        cs.push_back(256 + n);
        n++;
      }
    }

    for (size_t i = 0; i < bs.size(); i++) {
      uint8_t byte = (uint8_t)bs[i];
      uint32_t uni = (uint32_t)cs[i];

      unicode_to_byte[uni] = byte;

      std::string uni_utf8;
      append_utf8(uni_utf8, uni);
      byte_to_unicode_utf8[byte] = uni_utf8;
    }
  }

  bool load_vocab_json(const char* vocab_path) {
    std::ifstream f(vocab_path);
    if (!f.is_open()) return false;

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string content = buffer.str();

    id_to_token.clear();
    token_to_id.clear();

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

      std::string decoded;
      decoded.reserve(raw.size());

      for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
          char esc = raw[i + 1];
          if (esc == 'u' && i + 5 < raw.size()) {
            int cp = (hex_val(raw[i+2]) << 12) | (hex_val(raw[i+3]) << 8) |
                     (hex_val(raw[i+4]) << 4)  |  hex_val(raw[i+5]);
            append_utf8_cp(decoded, cp);
            i += 5;
          } else {
            if      (esc == 'n') decoded.push_back('\n');
            else if (esc == 't') decoded.push_back('\t');
            else if (esc == 'r') decoded.push_back('\r');
            else decoded.push_back(esc);
            i += 1;
          }
        } else {
          decoded.push_back(raw[i]);
        }
      }

      id_to_token[id] = decoded;
      token_to_id[decoded] = id;

      pos = val_end + 1;
    }

    return !id_to_token.empty();
  }

  bool load_merges_txt(const char* merges_path) {
    std::ifstream f(merges_path);
    if (!f.is_open()) return false;

    bpe_ranks.clear();
    bpe_cache.clear();

    std::string line;
    int rank = 0;

    while (std::getline(f, line)) {
      if (line.empty()) continue;
      if (!line.empty() && line[0] == '#') continue;

      size_t sp = line.find(' ');
      if (sp == std::string::npos) sp = line.find('\t');
      if (sp == std::string::npos) continue;

      std::string a = line.substr(0, sp);
      while (sp < line.size() && (line[sp] == ' ' || line[sp] == '\t')) sp++;
      if (sp >= line.size()) continue;
      std::string b = line.substr(sp);

      while (!b.empty() && (b.back() == ' ' || b.back() == '\t' || b.back() == '\r')) b.pop_back();
      if (a.empty() || b.empty()) continue;

      bpe_ranks[pair_key(a, b)] = rank++;
    }

    return !bpe_ranks.empty();
  }

  bool load(const char* vocab_json_path, const char* merges_txt_path) {
    init_gpt2_byte_maps();
    if (!load_vocab_json(vocab_json_path)) {
      std::printf("[tokenizer] Failed to load vocab.json\n");
      return false;
    }
    if (!load_merges_txt(merges_txt_path)) {
      std::printf("[tokenizer] Failed to load merges.txt\n");
      return false;
    }
    return true;
  }

  static inline std::string pair_key(const std::string& a, const std::string& b) {
    return a + '\n' + b;
  }

  int get_rank_or_inf(const std::string& a, const std::string& b) const {
    auto it = bpe_ranks.find(pair_key(a, b));
    if (it == bpe_ranks.end()) return 1 << 30;
    return it->second;
  }

  std::string bpe(const std::string& token) {
    auto itc = bpe_cache.find(token);
    if (itc != bpe_cache.end()) return itc->second;

    std::vector<std::string> word = split_utf8_codepoints(token);
    if (word.size() <= 1) {
      bpe_cache[token] = token;
      return token;
    }

    while (true) {
      int best_rank = 1 << 30;
      size_t best_i = (size_t)-1;

      for (size_t i = 0; i + 1 < word.size(); i++) {
        int r = get_rank_or_inf(word[i], word[i + 1]);
        if (r < best_rank) {
          best_rank = r;
          best_i = i;
        }
      }

      if (best_i == (size_t)-1 || best_rank == (1 << 30)) {
        break;
      }

      std::vector<std::string> new_word;
      new_word.reserve(word.size());

      for (size_t i = 0; i < word.size(); ) {
        if (i + 1 < word.size() && word[i] == word[best_i] && word[i + 1] == word[best_i + 1]) {
          new_word.push_back(word[i] + word[i + 1]);
          i += 2;
        } else {
          new_word.push_back(word[i]);
          i += 1;
        }
      }

      word.swap(new_word);
      if (word.size() <= 1) break;
    }

    std::string out = join_with_space(word);
    bpe_cache[token] = out;
    return out;
  }

  std::string byte_encode(const std::string& piece) const {
    std::string out;
    out.reserve(piece.size() * 2);

    for (unsigned char c : piece) {
      out += byte_to_unicode_utf8[(uint8_t)c];
    }
    return out;
  }

  std::string byte_decode(const std::string& byte_unicode) const {
    std::string bytes;
    bytes.reserve(byte_unicode.size());

    size_t i = 0;
    uint32_t cp = 0;
    while (next_codepoint_utf8(byte_unicode, i, cp)) {
      auto it = unicode_to_byte.find(cp);
      if (it != unicode_to_byte.end()) {
        bytes.push_back((char)it->second);
      } else if (cp <= 0xFF) {
        bytes.push_back((char)cp);
      } else {
        append_utf8(bytes, cp);
      }
    }
    return bytes;
  }

  static std::vector<std::string> gpt2_pretokenize_ascii(const std::string& text) {
    static const std::regex re(
      R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+)",
      std::regex::ECMAScript
    );

    std::vector<std::string> pieces;
    pieces.reserve(text.size() / 2 + 1);

    for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) {
      pieces.push_back(it->str());
    }
    return pieces;
  }

  std::vector<int> encode(const std::string& text) {
    std::vector<int> ids;
    ids.reserve(text.size() / 2 + 8);

    std::vector<std::string> pieces = gpt2_pretokenize_ascii(text);

    for (const std::string& piece : pieces) {
      std::string byte_encoded = byte_encode(piece);

      std::string bpe_out = bpe(byte_encoded);

      size_t start = 0;
      while (start < bpe_out.size()) {
        size_t sp = bpe_out.find(' ', start);
        if (sp == std::string::npos) sp = bpe_out.size();
        std::string tok = bpe_out.substr(start, sp - start);
        start = sp + 1;

        auto it = token_to_id.find(tok);
        if (it == token_to_id.end()) {
          std::printf("[tokenizer][WARN] Unknown BPE token (len=%zu)\n", tok.size());
          continue;
        }
        ids.push_back(it->second);
      }
    }

    return ids;
  }

  std::string decode(int id) const {
    auto it = id_to_token.find(id);
    if (it == id_to_token.end()) return "";
    return byte_decode(it->second);
  }

  std::string decode(const std::vector<int>& ids) const {
    std::string byte_unicode;
    byte_unicode.reserve(ids.size() * 4);

    for (int id : ids) {
      auto it = id_to_token.find(id);
      if (it != id_to_token.end()) byte_unicode += it->second;
    }
    return byte_decode(byte_unicode);
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
