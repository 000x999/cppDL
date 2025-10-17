#ifndef ALLOCATORBASE_HPP
#define ALLOCATORBASE_HPP 
#include <cstddef>

class allocator_base{
protected: 
  std::size_t m_total_alloc_size; 
  std::size_t m_used_allocs; 
  std::size_t m_peak_allocs;
public: 
  allocator_base(const std::size_t total_alloc_size)
    : m_total_alloc_size(total_alloc_size),
      m_used_allocs(0), 
      m_peak_allocs(0){}

  virtual ~allocator_base(){m_used_allocs = 0;}
  virtual void *allocate_memory(const std::size_t alloc_size, const std::size_t alloc_alignment = 0) = 0;  
  virtual void  free_memory(void *alloc_ptr) = 0;
  virtual void init_allocator() = 0;

  static const std::size_t alloc_padding(const std::size_t alloc_addr, const std::size_t alloc_alignment){
    const std::size_t alloc_multiplier = (alloc_addr / alloc_alignment) + 1;
    const std::size_t aligned_alloc_addr = alloc_multiplier * alloc_alignment;
    const std::size_t addr_padding = aligned_alloc_addr - alloc_addr; 
    return addr_padding;
  }

  static const std::size_t header_alloc_padding(const std::size_t alloc_addr, const std::size_t alloc_alignment, const std::size_t alloc_header){
    std::size_t addr_padding = allocator_base::alloc_padding(alloc_addr, alloc_alignment);
    std::size_t addr_space   = alloc_header;
    if(addr_padding < addr_space){
      addr_space -= addr_padding;
      if(addr_space % alloc_alignment < 0){
        addr_padding += alloc_alignment * (1 + (addr_space / alloc_alignment)); 
      }else{
        addr_padding += alloc_alignment * (    (addr_space / alloc_alignment)); 
      }
    }
    return addr_padding;     
  }

}; 

#endif 
