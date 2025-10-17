## The purpose of `platform_base` 
- The main purpose of the platform base interface is to create **platform agnostic** methods. That is, code that works regardless of what platform it's being executed on (MacOS, Linux, Windows, etc.)
- To achieve this, we create a `platform_base` interface on which platform specific code will inherit methods from and have custom implementations.
## `platform_base` Implementation  
```c++ 
struct platform_base_state;

typedef struct platform_base_config{
	const char *platform_application_name; 
}platform_base_config; 

typedef enum platform_base_error_code{
	PLATFORM_ERROR_SUCCESS        = 0, 
	PLATFORM_ERROR_UNKNOWN        = 1, 
	PLATFORM_ERROR_FILE_NOT_FOUND = 2, 
	PLATFORM_ERROR_FILE_LOCKED    = 3, 
	PLATFORM_ERROR_FILE_EXISTS    = 4   
}platform_base_error_code; 

class platform_base{
public:
	virtual ~platform_base() = default; 
	CPPDL_API virtual bool platform_base_startup(uint64_t memory_alloc, 
 	                        struct platform_base_state *base_state, 
 	                        platform_base_config *base_config) = 0;
 	                        
	CPPDL_API virtual void platform_base_shutdown(struct platform_base_state
	                        *base_state) = 0; 
	
	CPPDL_API virtual void *platform_base_allocate_memory()(uint64_t                                              alloc_size, bool is_aligned) = 0; 
							
	CPPDL_API virtual void platform_base_free_memory(void *memory_block,
							bool is_aligned);
	 
	CPPDL_API virtual void *platform_base_zero_memory(void *memory_block,
	                        uint64_t alloc_size) = 0;  
	                        
	virtual void *platform_base_copy_memory(void *memory_destination, 
	                        const void *memory_source,  uint64_t alloc_size) = 0;
	                         
	CPPDL_API virtual void *platform_base_set_memory(void *memory_destination,
	                        int32_t memory_value, uint64_t alloc_size) = 0; 

	CPPDL_API virtual void platform_base_console_write(struct platform_base_state 
								*base_state, cppdl_log_level log_level, const char
								*message_in) = 0; 
	CPPDL_API virtual float platform_base_time() = 0; 
	
	CPPDL_API virtual int32_t platform_base_get_processor_count() = 0;
	
	CPPDL_API virtual void platform_base_get_handle_info(uint64_t handle_size, 
	                        void *memory_size) = 0; 
}; 
```
- `platform_base_startup()` :  Will perform a boot routine within the `platform_base` layer. This startup function should be called twice, once to obtain the memory requirements and then a second time whilst passing an allocated block of memory to the `platform_base_state`. 
	- `memory_alloc`: A pointer to hold the byte size memory requirement
	- `base_state` : A pointer to a block of memory to hold the platforms' base state. 
	- `base_config`: A pointer to a `platform_base_config` structure required by the system (System dependent) 
	- Returns true if startup is successful. 
- `platform_base_shutdown()` : Simply shuts down the `platform_base_state` layer. 
- `platform_base_allocate_memory()`: Platform specific memory allocation routine using a given size. 
	- `alloc_size`: Size of the memory allocation in bytes. 
	- `is_aligned`: Specifier if the memory allocation should be aligned or not. 
	- Returns a pointer to the block of allocated memory 
- `platform_base_free_memory()`: Frees a specified block of memory 
	- `memory_block`:  Specific block of memory that needs to be freed. 
	- `is_aligned`: Boolean flag specifier to see if the block that needs to be freed is aligned or not. 
	- Returns true if freeing is successful. 
- `platform_base_zero_memory()`: Zeroes out a specified block of memory using platform-specific code. 
	- `memory_block`: The block of memory to be zeroed.
	- `alloc_size`: Size of the memory block to be zeroed. 
	- Returns a pointer to the block of zeroed memory. 
- `platform_base_copy_memory()`: Copies the memory from the passed in source, to the destination  block with a size specifier .
	- `memory_destination`: Destination address of the memory to be copied. 
	- `memory_source`: Source address of the memory to be copied. 
	- `alloc_size`: Size of the memory to be copied. 
	- Returns a pointer to the destination of the copied memory. 
- `platform_base_set_memory()`: Sets a block of memory using the passed in value. 
	- `memory_destination`: Destination address of the memory to be set. 
	- `memory_value`: Value of the memory to be set. 
	- `alloc_size`: Size of the data to be set. 
	- Returns a pointer to the destination of the set memory. 
- `platform_base_console_write()`: Platform specific log level based console output. 
	- `base_state`: Pointer to the platform's state instance. 
	- `message_in`: Messaged to be written to the console. 
	- `log_level`: Importance level of the message. 
- `platform_base_get_handle_info()`: Fetches memory amounts for platform specific handle data, can obtain a copy of that data. It needs to be called twice, once with `memory_size` = 0 to obtain the size and a second time where `memory_size` = `handle_size`. 
	- `handle_size` : A pointer to hold the full memory requirements. 
	- `memory_size`: Size of the block of memory. 