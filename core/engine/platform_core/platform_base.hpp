#ifndef PLATFORMBASE_HPP
#define PLATFORMBASE_HPP
#include "../../defines.h"
#include "../../include/logger_core/logger.h"
#include <cstdint>

class platform_base{
public: 
  struct platform_base_state; 
  struct platform_base_config{
    const char *platform_application_name; 
  };
  enum platform_base_error_code{
    PLATFORM_ERROR_SUCCESS     = 0, 
    PLATFORM_ERROR_UNKNOWN     = 1,
    PLATFORM_ERROR_FNF         = 2, 
    PLATFORM_ERROR_FILE_LOCKED = 3,
    PLATFORM_ERROR_FILE_EXISTS = 4
  };
  virtual ~platform_base() = default; 
  CPPDL_API virtual bool platform_base_startup (uint64_t memory_alloc, struct platform_base_state *base_state, platform_base_config *base_config) = 0; 
  CPPDL_API virtual void platform_base_shutdown(struct platform_base_state *base_state) = 0; 
  CPPDL_API virtual void *platform_base_allocate_memory(uint64_t alloc_size, bool is_aligned) = 0;
  CPPDL_API virtual void platform_base_free_memory(void *memory_block, bool is_aligned) = 0; 
  CPPDL_API virtual void *platform_base_copy_memory(void *memory_destination, const void *memory_source, uint64_t alloc_size) = 0;
  CPPDL_API virtual void *platform_base_set_memory(void *memory_destination, int32_t memory_value, uint64_t alloc_size) = 0;
  CPPDL_API virtual void platform_base_console_write(struct platform_base_state *base_state, cppdl_log_level log_level, const char *message_in) = 0; 
  CPPDL_API virtual float platform_base_get_time() = 0; 
  CPPDL_API virtual int32_t platform_base_get_processor_count() = 0; 
  CPPDL_API virtual void platform_base_get_handle_info(uint64_t handle_size, void *memory_size) = 0; 
};
#endif
