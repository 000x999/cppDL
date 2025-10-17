## What is the `alloc_list` 
- The `alloc_list` structure is essentially based around free lists. Free lists are data structures used in dynamic memory allocation schemes. It connects unallocated regions of memory together using a linked list. The first word of each unallocated region is used as a pointer to the next one.
## `alloc_list` Implementation 
```c++
typedef struct alloc_list_state{
	void *list_memory; 
}alloc_list_state; 

class alloc_list{
public: 
	CPPDL_API void init_alloc_list(uint64_t alloc_size, uint64_t *memory_size, void
						*list_memory, alloc_list_state *out_list); 
						
	CPPDL_API void destroy_alloc_list(alloc_list_state *list_in);
	 
	CPPDL_API bool alloc_list_block(alloc_list_state *list_in, uint64_t block_size,
						uint64_t *alloc_offset); 
	                    
	CPPDL_API bool alloc_list_free_block(alloc_list_state *list_in, uint64_t
	                    block_size, uint64_t block_offset); 
						
	CPPDL_API bool resize_alloc_list(alloc_list_state *list_in, uint64_t
	                    *memory_size, void *new_alloc_memory, uint64_t
	                    new_alloc_size, void **old_alloc_memory);
						 
	CPPDL_API void clear_alloc_list(alloc_list_state *list_in);
	 
	CPPDL_API uint64_t alloc_list_free(alloc_list_state *list_in); 
}; 
```
- `init_alloc_list()`: Will create a new `alloc_list` or obtain the memory required for one. This needs to be called twice, once for passing 0 to memory and fetching the memory requirements and another time for passing an allocated block to memory. 
	- `alloc_size`: Total size in bytes that the `alloc_list` should keep track of. 
	- `memory_size`: A pointer to hold the total memory requirement for the `alloc_list`
	- `list_memory`: A pre-allocated block of memory for the `alloc_list` to use. 
	- `out_list`: A pointer to hold the `alloc_list_state`. 
- `destroy_alloc_list()`: Destroys a passed in `alloc_list_state` 
- `alloc_list_block()`: 