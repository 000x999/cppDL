#ifndef ALLOCLIST_HPP
#define ALLOCLIST_HPP 
#include <limits>
#include <algorithm>

#include "allocator_base.hpp"
#include "../../defines.h" 
#include "data_structures/linked_list.hpp"
#include "../../include/logger_core/logger.h"

class alloc_list : public allocator_base{
public: 
  enum alloc_policy{
    FIND_FIRST, 
    FIND_BEST
  };

private: 
  struct list_header {std::size_t alloc_list_block_size;}; 
  struct alloc_header{std::size_t header_size; std::size_t header_padding;};
  using  alloc_list_node = linked_list<list_header>::list_node;
  void   *m_alloc_ptr    = nullptr; 
  alloc_policy m_alloc_policy; 
  linked_list<list_header> m_alloc_list;

public:
  alloc_list(const std::size_t alloc_size, const alloc_policy policy_type) : allocator_base(alloc_size), m_alloc_policy(policy_type) {}
  
  ~alloc_list(){
    free(m_alloc_ptr); 
    m_alloc_ptr = nullptr; 
  }
  
  void *allocate_memory(const std::size_t alloc_size, const std::size_t alloc_alignment = 0) override {
    const std::size_t alloc_header_size = sizeof(alloc_header);
    const std::size_t free_header_size  = sizeof(list_header);
    CPPDL_WARN("ALLOCATION SIZE MUST BE BIGGER THAN: ", sizeof(alloc_list_node));
    CPPDL_WARN("MEMORY ALIGNMENT MUST BE AT LEAST 8, CURRENTLY: ", (float)alloc_alignment);
    assert(alloc_size >= sizeof(alloc_list_node)); 
    assert(alloc_alignment >= 8);

    std::size_t alloc_list_padding; 
    alloc_list_node *affected_node; 
    alloc_list_node *previous_node;
    this->find_block(alloc_size, alloc_alignment, alloc_list_padding, previous_node, affected_node);
    CPPDL_WARN("AFFECTED NODE MUST NOT BE NULL"); 
    assert(affected_node != nullptr);

    const std::size_t alloc_alignment_padding = alloc_list_padding - alloc_header_size; 
    const std::size_t required_alloc_size     = alloc_size + alloc_list_padding; 
    const std::size_t alloc_remainder         = affected_node->node_data.alloc_list_block_size - required_alloc_size;
    if(alloc_remainder > 0){
      alloc_list_node *next_free_node = (alloc_list_node*)((std::size_t)affected_node + required_alloc_size);
      next_free_node->node_data.alloc_list_block_size = alloc_remainder; 
      m_alloc_list.insert_node(affected_node, next_free_node);
    }
    m_alloc_list.pop_node(previous_node, affected_node); 
    const std::size_t alloc_header_addr = (std::size_t)affected_node + alloc_alignment_padding; 
    const std::size_t alloc_data_addr   = alloc_header_addr + alloc_header_size;
    m_used_allocs += required_alloc_size; 
    m_peak_allocs  = std::max(m_peak_allocs, m_used_allocs);

    return (void*)alloc_data_addr;
  }
  
  void free_memory(void *alloc_ptr) override{
    const std::size_t current_addr = (std::size_t)alloc_ptr; 
    const std::size_t header_addr  = current_addr - sizeof(alloc_header);
    const alloc_header *current_alloc_header{(alloc_header*)header_addr};
    alloc_list_node *free_node = (alloc_list_node*)header_addr;
    free_node->node_data.alloc_list_block_size = current_alloc_header->header_size + current_alloc_header->header_padding;
    free_node->next_node = nullptr;

    alloc_list_node *block_iter = m_alloc_list.head_node;
    alloc_list_node *prev_block_iter = nullptr; 
    while(block_iter != nullptr){
      if(alloc_ptr < block_iter){
        m_alloc_list.insert_node(prev_block_iter, free_node);
        break;
      }
      prev_block_iter = block_iter; 
      block_iter = block_iter->next_node;
    }
    m_used_allocs -= free_node->node_data.alloc_list_block_size;
    alloc_list::alloc_list_merge(prev_block_iter, free_node);
  }
  
  void init_allocator() override {
    if(m_alloc_ptr != nullptr){
      free(m_alloc_ptr); 
      m_alloc_ptr = nullptr; 
    }
    m_alloc_ptr = malloc(m_total_alloc_size);
    m_used_allocs = 0; 
    m_peak_allocs = 0; 
    alloc_list_node *first_node = (alloc_list_node*)m_alloc_ptr; 
    first_node->node_data.alloc_list_block_size = m_total_alloc_size; 
    first_node->next_node  = nullptr; 
    m_alloc_list.head_node = nullptr;
    m_alloc_list.insert_node(nullptr, first_node);     
  }

  void reset_memory(){
    m_used_allocs = 0; 
    m_peak_allocs = 0; 
    alloc_list_node *first_node = (alloc_list_node*)m_alloc_ptr; 
    first_node->node_data.alloc_list_block_size = m_total_alloc_size; 
    first_node->next_node  = nullptr; 
    m_alloc_list.head_node = nullptr;
    m_alloc_list.insert_node(nullptr, first_node);     
  }

private: 
  alloc_list(alloc_list &alloc_list_in);

  void alloc_list_merge(alloc_list_node *previous_block, alloc_list_node *free_block){
    if((free_block->next_node != nullptr) && (std::size_t)free_block + free_block->node_data.alloc_list_block_size == (std::size_t)free_block->next_node){
      free_block->node_data.alloc_list_block_size += free_block->next_node->node_data.alloc_list_block_size; 
      m_alloc_list.pop_node(free_block, free_block->next_node);
    }
    if(previous_block != nullptr && (std::size_t)previous_block + previous_block->node_data.alloc_list_block_size == (std::size_t)free_block){
      previous_block->node_data.alloc_list_block_size += free_block->node_data.alloc_list_block_size; 
      m_alloc_list.pop_node(previous_block, free_block); 
    }
  }
  
  void find_block(const std::size_t alloc_size, const std::size_t alloc_alignment, std::size_t &alloc_padding, alloc_list_node *&previous_node, alloc_list_node *&found_node){
    switch(m_alloc_policy){
      case FIND_FIRST:
        find_first_block(alloc_size, alloc_alignment, alloc_padding, previous_node, found_node); 
        break; 
      case FIND_BEST:
        find_best_block(alloc_size, alloc_alignment,  alloc_padding, previous_node, found_node); 
    } 
  }
  
  void find_best_block(const std::size_t alloc_size, const std::size_t alloc_alignment, std::size_t &alloc_padding, alloc_list_node *&previous_node, alloc_list_node *&found_node){
    std::size_t alloc_diff = std::numeric_limits<std::size_t>::max();
    alloc_list_node *best_alloc_block = nullptr; 
    alloc_list_node *block_iter = m_alloc_list.head_node; 
    alloc_list_node *prev_block_iter = nullptr; 
    while(block_iter != nullptr){
      alloc_padding = allocator_base::header_alloc_padding((std::size_t)block_iter, alloc_alignment, sizeof(alloc_header));
      const std::size_t required_alloc_space = alloc_size + alloc_padding; 
      if(block_iter->node_data.alloc_list_block_size >= required_alloc_space && (block_iter->node_data.alloc_list_block_size - required_alloc_space < alloc_diff)){
        best_alloc_block = block_iter; 
      }
      prev_block_iter = block_iter; 
      block_iter = block_iter->next_node; 
    }
    previous_node = prev_block_iter; 
    found_node = best_alloc_block; 
  }
  
  void find_first_block(const std::size_t alloc_size, const std::size_t alloc_alignment, std::size_t &alloc_padding, alloc_list_node *&previous_node, alloc_list_node *&found_node){
    alloc_list_node *block_iter = m_alloc_list.head_node; 
    alloc_list_node *prev_block_iter = nullptr;
    while(block_iter != nullptr){
      alloc_padding = allocator_base::header_alloc_padding((std::size_t)block_iter, alloc_alignment, sizeof(alloc_header));
      const std::size_t required_alloc_space = alloc_size + alloc_padding; 
      if(block_iter->node_data.alloc_list_block_size >= required_alloc_space){
        break; 
      }
      prev_block_iter = block_iter; 
      block_iter = block_iter->next_node; 
    }
    previous_node = prev_block_iter;
    found_node = block_iter; 
  }

};
#endif 
