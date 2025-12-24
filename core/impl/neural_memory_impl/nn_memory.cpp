#include "neural_memory/nn_memory.hpp"

size_t memory::neural_arena::align_forward(size_t ptr_address){
 if(ptr_address % nn_align == 0){
    return ptr_address;
  }else{
    return (ptr_address + (nn_align - (ptr_address % nn_align))); 
  }
}

memory::neural_arena::neural_arena(size_t nn_arena_size)
:
  nn_buffer(new char[nn_arena_size]), 
  nn_capacity(nn_arena_size), 
  nn_offset(0)
{}

void   memory::neural_arena::nn_reset       () { nn_offset = 0;                    }
size_t memory::neural_arena::nn_used        () { return nn_offset;                 }
size_t memory::neural_arena::nn_remaining   () { return (nn_capacity - nn_offset); }
