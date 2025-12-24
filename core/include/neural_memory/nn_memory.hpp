#ifndef NN_MEMORY_HPP
#define NN_MEMORY_HPP
#include <cstddef> 
#include <memory> 
#include <iostream>
#include <new>
#include "cppdl_defines.h"

namespace memory{
class neural_arena{
private: 
  static constexpr size_t nn_align  = 32; 
  size_t                  nn_offset;
  size_t                  nn_capacity;
  std::unique_ptr<char[]> nn_buffer;

  size_t align_forward( size_t ptr_address );
  
public:
  explicit neural_arena( size_t nn_arena_size );

  neural_arena           ( const neural_arena& )  = delete;
  neural_arena operator= ( const neural_arena& )  = delete;

  template<typename T> 
  T*     nn_alloc     ( size_t nn_alloc_size ){
    uintptr_t nn_addr      = reinterpret_cast<uintptr_t>(nn_buffer.get()); 
    uintptr_t current_addr = nn_addr + nn_offset; 

    size_t    padding      = 0;

    if(current_addr % nn_align != 0){
      padding = nn_align - (current_addr % nn_align); 
    }

    size_t bytes_needed = (nn_alloc_size * sizeof(T)); 
    if(nn_offset + padding + bytes_needed > nn_capacity){
      std::cout << "arena size: " << nn_capacity<< '\n';
      std::cout << "total needed size: " << nn_offset + padding + bytes_needed << '\n'; 
      throw std::bad_alloc(); 
    }
    T* nn_aligned_ptr = reinterpret_cast<T*>(nn_buffer.get() + nn_offset + padding); 
    nn_offset        += padding + bytes_needed;

    return nn_aligned_ptr; 
  }

  void   nn_reset     ();
  size_t nn_used      ();
  size_t nn_remaining ();
};
};
#endif 

