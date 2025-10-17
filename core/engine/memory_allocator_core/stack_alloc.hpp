#ifndef STACKALLOC_HPP
#define STACKALLOC_HPP 
#include "allocator_base.hpp"
#include "../../defines.h"
#include "../../include/logger_core/logger.h"
#include <stdlib.h>
#include <algorithm>

class stack_alloc : public allocator_base{
private:
  stack_alloc(stack_alloc &stack_allocator);
  struct stack_header{ std::size_t stack_addr_padding; };

protected:
  void *m_alloc_ptr = nullptr; 
  std::size_t m_alloc_offset;

public:
  stack_alloc(const std::size_t alloc_size): allocator_base(alloc_size){}
  
  ~stack_alloc() override{
    free(m_alloc_ptr); 
    m_alloc_ptr = nullptr;
  }

  void *allocate_memory(const std::size_t alloc_size, const std::size_t alloc_alignment = 0) override{
    const std::size_t current_addr = (std::size_t)m_alloc_ptr + m_alloc_offset;
    std::size_t addr_padding = allocator_base::header_alloc_padding(current_addr, alloc_alignment, sizeof(stack_header));
    
    if(m_alloc_offset + addr_padding + alloc_size > m_total_alloc_size){
      return nullptr;
    }
    
    m_alloc_offset += addr_padding; 
    const std::size_t next_addr   = current_addr + m_alloc_offset;
    const std::size_t header_addr = next_addr - sizeof(stack_header);
    stack_header current_addr_header {addr_padding};
    stack_header *header_ptr = (stack_header*) header_addr;
    *header_ptr = current_addr_header;
    
    m_alloc_offset += alloc_size;
    m_used_allocs = m_alloc_offset, 
    m_peak_allocs = std::max(m_peak_allocs, m_used_allocs); 
    return (void*) next_addr; 
  }

  void free_memory(void *alloc_ptr) override{
    const std::size_t current_addr  = (std::size_t)alloc_ptr;
    const std::size_t previous_addr = current_addr - sizeof(stack_header);
    const stack_header *current_addr_header {(stack_header*) previous_addr};

    m_alloc_offset = current_addr - current_addr_header->stack_addr_padding - (std::size_t)m_alloc_ptr;
    m_used_allocs  = m_alloc_offset;

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
