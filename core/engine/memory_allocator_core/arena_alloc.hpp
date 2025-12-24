#ifndef ARENAALLOC_HPP
#define ARENAALLOC_HPP
#include "allocator_base.hpp"
#include "data_structures/stacked_list.hpp"
#include "../../defines.h"
#include "../../include/logger_core/logger.hpp"
#include <algorithm>

class arena_alloc : public allocator_base{
private:
  struct arena_header{}; 
  using arena_node = stacked_list<arena_header>::list_node;
  stacked_list<arena_header> m_arena_list;
  void *m_alloc_ptr = nullptr; 
  std::size_t m_arena_chunk_size;
public:
  arena_alloc(const std::size_t alloc_size, const std::size_t arena_chunk_size):allocator_base(alloc_size){
    CPPDL_WARN("CHUNK SIZE MUST BE GREATER OR EQUAL TO 8"); 
    CPPDL_WARN("TOTAL SIZE MUST BE A MULTIPLE OF 8");
    this->m_arena_chunk_size = arena_chunk_size; 
  }
  virtual ~arena_alloc(){free(m_alloc_ptr);} 
  
  void *allocate_memory(const std::size_t alloc_size, const std::size_t alloc_alignment) override{
    CPPDL_WARN("ALLOCATION SIZES MUST BE EQUAL TO THE CHUNK SIZES"); 
    assert(alloc_size == this->m_arena_chunk_size);
    arena_node *free_chunk = m_arena_list.pop_node();
    CPPDL_WARN("ARENA IS FULL"); 
    assert(free_chunk != nullptr);
    m_used_allocs += m_arena_chunk_size; 
    m_peak_allocs = std::max(m_peak_allocs, m_used_allocs); 
    return (void*)free_chunk; 
  }
  
  void free_memory(void *alloc_ptr) override{
    m_used_allocs -= m_arena_chunk_size; 
    m_arena_list.push_node((arena_node*)alloc_ptr);
  }
  
  void init_allocator() override{
    m_alloc_ptr = malloc(m_total_alloc_size);
    m_used_allocs = 0; 
    m_peak_allocs = 0; 
    const int arena_chunks = m_total_alloc_size / m_arena_chunk_size;
    for(int i = 0; i < arena_chunks; ++i){
      std::size_t chunk_addr = (std::size_t)m_alloc_ptr + i * m_arena_chunk_size; 
      m_arena_list.push_node((arena_node*)chunk_addr); 
    }
  }

  void arena_reset(){
    m_used_allocs = 0; 
    m_peak_allocs = 0; 
    const int arena_chunks = m_total_alloc_size / m_arena_chunk_size;
    for(int i = 0; i < arena_chunks; ++i){
      std::size_t chunk_addr = (std::size_t)m_alloc_ptr + i * m_arena_chunk_size; 
      m_arena_list.push_node((arena_node*)chunk_addr); 
    }
  }

private:
  arena_alloc(arena_alloc &arena_in);

};
#endif 
