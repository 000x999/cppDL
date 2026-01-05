#ifndef SAFETENSOR_READER_H
#define SAFETENSOR_READER_H
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace safetensor {
constexpr size_t MAX_TENSORS   = 512;
constexpr size_t MAX_NAME_LEN  = 256;
constexpr size_t MAX_DIMS      = 8;

struct tensor_entry {
  char   name[MAX_NAME_LEN];
  size_t shape[MAX_DIMS];
  size_t strides[MAX_DIMS];
  int    ndim;
  size_t data_offset;
  size_t data_size;
  char   dtype[16];
};

struct safetensor_file {
  char*        file_data;
  size_t       file_size;
  char*        tensor_data_start;
  tensor_entry entries[MAX_TENSORS];
  size_t       num_entries;
};

size_t        parse_json_number (const char* json, size_t* cursor                            );
size_t        parse_json_string (const char* json, size_t* cursor, char* out, size_t max_len );
void          skip_ws           (const char* json, size_t* cursor                            );
size_t        find_char         (const char* json, size_t  cursor, char c                    );
bool          parse_header      (const char* json, size_t json_len, safetensor_file* sf      );
bool          load_safetensor   (const char* path, safetensor_file* sf                       );
void          free_safetensor   (safetensor_file* sf                                         );
tensor_entry* find_entry        (safetensor_file* sf, const char* name                       );
float*        get_tensor_data   (safetensor_file* sf, const char* name                       );
void          print_entries     (safetensor_file* sf                                         );

}  // namespace safetensor
#endif 
