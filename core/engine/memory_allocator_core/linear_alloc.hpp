#ifndef LINEARALLOC_HPP
#define LINEARALLOC_HPP 
#include "../../defines.h"
#include "../../include/logger_core/logger.hpp"
#include "allocator_base.hpp"
#include <stdlib.h> 
#include <algorithm>

class linear_alloc : public allocator_base{
private:
  linear_alloc(linear_alloc &linear_allocator); 

protected: 
  void *m_alloc_ptr = nullptr;
  std::size_t m_alloc_offset;

public: 
  linear_alloc(const std::size_t alloc_size): allocator_base(alloc_size){}

  ~linear_alloc() override {
    free(m_alloc_ptr);
    m_alloc_ptr = nullptr; 
  }

  void *allocate_memory(const std::size_t alloc_size, const std::size_t alloc_alignment) override{
    std::size_t alloc_padding = 0; 
    std::size_t addr_padding  = 0;
    const std::size_t current_addr = (std::size_t)m_alloc_ptr + m_alloc_offset;
    if(alloc_alignment != 0 && m_alloc_offset % alloc_alignment != 0){
      alloc_padding = allocator_base::alloc_padding(current_addr, alloc_alignment); 
    }
    if(m_alloc_offset + alloc_padding + alloc_size > m_total_alloc_size){
      return nullptr; 
    }
    m_alloc_offset += addr_padding; 
    const std::size_t next_addr = current_addr + addr_padding; 
    m_alloc_offset += alloc_size;
    m_used_allocs = m_alloc_offset; 
    m_peak_allocs = std::max(m_peak_allocs, m_used_allocs);
    return (void*) next_addr; 
  }

  void free_memory(void *alloc_ptr) override{
    CPPDL_ERROR("FREE IS N/A, USE RESET() METHOD INSTEAD");
  }

  void init_allocator() override{
    if(m_alloc_ptr != nullptr){
      free(m_alloc_ptr); 
    }
    m_alloc_ptr = malloc(m_total_alloc_size);
    m_alloc_offset = 0; 
  }

  void reset_allocator(){
    m_alloc_offset = 0; 
    m_used_allocs  = 0;
    m_peak_allocs  = 0; 
  }

};
#endif 
