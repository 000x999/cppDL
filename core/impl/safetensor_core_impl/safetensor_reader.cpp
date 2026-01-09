#include "safetensor_core/safetensor_reader.hpp"

size_t safetensor::parse_json_number(const char* json, size_t* cursor) {
  size_t val = 0;
  while (json[*cursor] >= '0' && json[*cursor] <= '9') {
    val = val * 10 + (json[*cursor] - '0');
    (*cursor)++;
  }
  return val;
}

size_t safetensor::parse_json_string(const char* json, size_t* cursor, char* out, size_t max_len) {
  (*cursor)++; 
  size_t len = 0;
  while (json[*cursor] != '"' && len < max_len - 1) {
    out[len++] = json[*cursor];
    (*cursor)++;
  }
  out[len] = '\0';
  (*cursor)++;
  return len;
}

void safetensor::skip_ws(const char* json, size_t* cursor) {
  while (json[*cursor] == ' ' || json[*cursor] == '\n' || 
    json[*cursor] == '\r' || json[*cursor] == '\t') {
    (*cursor)++;
  }
}

size_t safetensor::find_char(const char* json, size_t cursor, char c) {
  while (json[cursor] != c && json[cursor] != '\0') {
    cursor++;
  }
  return cursor;
}

bool safetensor::parse_header(const char* json, size_t json_len, safetensor::safetensor_file* sf) {
  size_t cursor = 0;
  sf->num_entries = 0;
  
  skip_ws(json, &cursor);
  if (json[cursor] != '{') return false;
  cursor++;
  
  while (cursor < json_len && sf->num_entries < MAX_TENSORS) {
    skip_ws(json, &cursor);
    
    if (json[cursor] == '}') { 
      break;
    }
    if (json[cursor] == ',') { 
      cursor++; continue; 
    }
    
    if (json[cursor] != '"') {
      return false;
    }
    
    char key_name[MAX_NAME_LEN];
    parse_json_string(json, &cursor, key_name, MAX_NAME_LEN);
    
    skip_ws(json, &cursor);
    if (json[cursor] != ':') {
      return false;
    }
    cursor++;
    skip_ws(json, &cursor);
    
    if (std::strcmp(key_name, "__metadata__") == 0) {
      if (json[cursor] != '{') { 
        return false;
      }
      cursor++;
      int brace_count = 1;
      while (cursor < json_len && brace_count > 0) {
        if (json[cursor] == '{') {
          brace_count++; 
        }
        else if (json[cursor] == '}'){
          brace_count--;
        }
        cursor++;
      }
      continue;
    }
    
    tensor_entry* entry = &sf->entries[sf->num_entries];
    std::strncpy(entry->name, key_name, MAX_NAME_LEN - 1);
    entry->name[MAX_NAME_LEN - 1] = '\0';
    
    if (json[cursor] != '{') return false;
    cursor++;
    
    entry->ndim = 0;
    entry->data_offset = 0;
    entry->data_size = 0;
    std::memset(entry->dtype, 0, sizeof(entry->dtype));
    std::memset(entry->shape, 0, sizeof(entry->shape));
      
    while (json[cursor] != '}' && cursor < json_len) {
      skip_ws(json, &cursor);
      if (json[cursor] == ',') {
        cursor++; continue; 
      }

      if (json[cursor] == '}'){ 
        break;
      }
      
      if (json[cursor] != '"') {
        return false;
      }
      char field_name[64];
      parse_json_string(json, &cursor, field_name, 64);
      
      skip_ws(json, &cursor);
      if (json[cursor] != ':') {
        return false;
      }
      cursor++;
      skip_ws(json, &cursor);
      
      if (std::strcmp(field_name, "dtype") == 0) {
        parse_json_string(json, &cursor, entry->dtype, 16);
      } else if (std::strcmp(field_name, "shape") == 0) {
          if (json[cursor] != '[') return false;
          cursor++;
          entry->ndim = 0;
          
          while (json[cursor] != ']' && cursor < json_len) {
            skip_ws(json, &cursor);
            if (json[cursor] == ',') { 
              cursor++; continue; 
            }

            if (json[cursor] == ']') {
              break;
            }

            if (json[cursor] >= '0' && json[cursor] <= '9') {
                entry->shape[entry->ndim++] = parse_json_number(json, &cursor);
            } else {
                cursor++;
            }
          }
          if (json[cursor] == ']'){
            cursor++;
          }
        }else if (std::strcmp(field_name, "data_offsets") == 0) {
          if (json[cursor] != '[') {
            return false;
          }
          cursor++;
          skip_ws(json, &cursor);
          size_t start = parse_json_number(json, &cursor);
          skip_ws(json, &cursor);
          if (json[cursor] == ',') {
            cursor++;
          }
          skip_ws(json, &cursor);
          size_t end = parse_json_number(json, &cursor);
          entry->data_offset = start;
          entry->data_size = end - start;
          while (json[cursor] != ']' && cursor < json_len) {
            cursor++;
          }
          if (json[cursor] == ']') {
            cursor++;
          }
        }
      }
      if (json[cursor] == '}') {
        cursor++;
      }
        
      if (entry->ndim > 0) {
        entry->strides[entry->ndim - 1] = 1;
        for (int i = entry->ndim - 2; i >= 0; i--) {
          entry->strides[i] = entry->strides[i + 1] * entry->shape[i + 1];
        }
      }
      
    sf->num_entries++;
  }
  return true;
}

bool safetensor::load_safetensor(const char* path, safetensor::safetensor_file* sf) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::printf("ERROR: Could not open file: %s\n", path);
    return false;
  }
  
  std::fseek(f, 0, SEEK_END);
  sf->file_size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  
  std::printf("DEBUG: File size: %zu bytes\n", sf->file_size);
  
  sf->file_data = (char*)std::malloc(sf->file_size);
  if (!sf->file_data) {
    std::fclose(f);
    return false;
  }
  
  if (std::fread(sf->file_data, 1, sf->file_size, f) != sf->file_size) {
    std::free(sf->file_data);
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  
  uint64_t header_size = 0;
  std::memcpy(&header_size, sf->file_data, sizeof(uint64_t));
  
  std::printf("DEBUG: Header size: %zu bytes\n", (size_t)header_size);
  
  const char* json_start = sf->file_data + 8;
  std::printf("DEBUG: Header start (first 500 chars):\n%.500s\n\n", json_start);
  
  if (!parse_header(json_start, header_size, sf)) {
    std::printf("ERROR: parse_header failed\n");
    std::free(sf->file_data);
    return false;
  }
  
  sf->tensor_data_start = sf->file_data + 8 + header_size;
  
  return true;
}

void safetensor::free_safetensor(safetensor::safetensor_file* sf) {
  if (sf->file_data) {
    std::free(sf->file_data);
    sf->file_data = nullptr;
  }
}

safetensor::tensor_entry* safetensor::find_entry(safetensor::safetensor_file* sf, const char* name) {
  for (size_t i = 0; i < sf->num_entries; i++) {
    if (std::strcmp(sf->entries[i].name, name) == 0) {
      return &sf->entries[i];
    }
  }
  return nullptr;
}

float* safetensor::get_tensor_data(safetensor::safetensor_file* sf, const char* name) {
  safetensor::tensor_entry* entry = safetensor::find_entry(sf, name);
  if (!entry) return nullptr;
  return reinterpret_cast<float*>(sf->tensor_data_start + entry->data_offset);
}

void safetensor::print_entries(safetensor::safetensor_file* sf) {
  for (size_t i = 0; i < sf->num_entries; i++) {
    safetensor::tensor_entry* e = &sf->entries[i];
    std::printf("%s [dtype=%s, shape=(", e->name, e->dtype);
    for (int d = 0; d < e->ndim; d++) {
      std::printf("%zu%s", e->shape[d], d < e->ndim - 1 ? ", " : "");
    }
    std::printf("), offset=%zu, size=%zu]\n", e->data_offset, e->data_size);
  }
}

