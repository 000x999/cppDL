#ifndef MEMORYCORE_HPP
#define MEMORYCORE_HPP 
#include "../../defines.h"
#include "../../include/logger_core/logger.h"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <string>

class memory_core{
private:
  enum memory_tag{
    MEMORY_TAG_UNKNOWN, 
    MEMORY_TAG_ARRAY,
    MEMORY_TAG_DARRAY,
    MEMORY_TAG_DICT,
    MEMORY_TAG_RING_QUEUE,
    MEMORY_TAG_BST,
    MEMORY_TAG_STRING,
    MEMORY_TAG_JOB,
    MEMORY_TAG_STATE,
    MEMORY_TAG_ENGINE,
    MEMORY_TAG_MATRIX,
    MEMORY_TAG_TENSOR,
    MEMORY_TAG_DATA_LOADER,
    MEMORY_TAG_NEURAL_NETWORK,
    MEMORY_TAG_PLATFORM,
    MEMORY_TAG_GPU, 
    MEMORY_TAG_MAX_TAGS
  }; 

  const char *memory_tag_types[MEMORY_TAG_MAX_TAGS] = { 
    "UNKNOWN          ", 
    "JOB              ",
    "STATE            ",
    "ENGINE           ",
    "MATRIX           ",
    "TENSOR           ",
    "DATA_LOADER      ",
    "NEURAL_NETWORK   ",
  };
 
  struct memory_core_config{
    size_t alloc_size; 
  }; 
  
  struct memory_core_stats{
    size_t total_core_allocs; 
    size_t tagged_core_allocs[MEMORY_TAG_MAX_TAGS]; 
    size_t new_core_tagged_allocs[MEMORY_TAG_MAX_TAGS]; 
    size_t new_core_tagged_deallocs[MEMORY_TAG_MAX_TAGS]; 
  }; 

  struct memory_core_state{
    memory_core_config state_config; 
    struct memory_core_stats state_stats; 
    size_t total_state_allocs;
    size_t state_alloc_requirement; 
    void *state_alloc_block; 
  };

public: 
  CPPDL_API static bool create_memory_core             (memory_core_config core_config); 
  CPPDL_API static void shutdown_memory_core           ();
  CPPDL_API static void *allocate_memory_core          (size_t size, memory_tag tag_type); 
  CPPDL_API static void *memory_core_allocate_aligned  (size_t alloc_size, size_t alloc_alignment, memory_tag tag_type);
  CPPDL_API static void memory_core_allocate_report    (size_t alloc_size, memory_tag tag_type);
  CPPDL_API static void memory_core_realloc            (void *block_size, size_t old_alloc_size, size_t new_alloc_size, memory_tag tag_type);
  CPPDL_API static void *memory_core_realloc_aligned   (void *block_size, size_t old_alloc_size, size_t new_alloc_size, size_t alloc_alignment, memory_tag tag_type); 
  CPPDL_API static void memory_core_realloc_report     (size_t old_alloc_size, size_t new_alloc_size, memory_tag tag_type); 
  CPPDL_API static void memory_core_free               (void *block_size, size_t alloc_size, memory_tag tag_type);
  CPPDL_API static void memory_core_free_aligned       (void *block_size, size_t alloc_size, size_t alloc_alignment, memory_tag tag_type); 
  CPPDL_API static void memory_core_free_report        (size_t alloc_size, memory_tag tag_type); 
  CPPDL_API static bool memory_core_get_alignment      (void *block_size, uintptr_t *alloc_size, uintptr_t *alloc_alignment); 
  CPPDL_API static void *memory_core_zero_memory       (void *block_size, size_t alloc_size); 
  CPPDL_API static void *memory_core_copy_memory       (void *memory_destination, const void *memory_source, size_t size);
  CPPDL_API static void *memory_core_set_memory        (void *memory_destination, size_t alloc_value, size_t alloc_size);
  CPPDL_API static std::string memory_core_debug_stats (); 
  CPPDL_API static size_t memory_core_alloc_count      (); 
}; 
#endif 
