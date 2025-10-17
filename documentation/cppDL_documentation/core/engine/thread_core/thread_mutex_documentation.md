## What is a `thread_mutex`
- Mutex's or Locks are synchronization primitives that prevent states from being modified or accessed by multiple execution threads at the same time. Locks enforce mutual exclusion concurrency control policies. 
## `thread_mutex` implementation 
```c++
class thread_mutex{
public: 
	void *internal_mutex_data; 
	//more next ... 
}
```
- Mutex is used to limit access to a resource when there are multiple threads of execution around that resource. 
```c++
	CPPDL_API bool thread_mutex_create(thread_mutex *output_mutex); 
```
- `thread_mutex_create()` creates a mutex, it takes in a pointer to hold the created mutex and returns true if created successfully. 
```c++
	CPPDL_API void thread_mutex_destroy(thread_mutex *mutex_in); 
```
- `thread_mutex_destroy()` destroys a mutex and takes in a pointer to a mutex that needs to be destroyed. 
```c++
	CPPDL_API bool thread_mutex_lock(thread_mutex *mutex_in); 
```
- `thread_mutex_lock()` creates a mutex lock, takes in a pointer to a mutex and returns true if the lock is successful. 
```c++
	CPPDL_API bool thread_mutex_unlock(thread_mutex *mutex_in); 
```
- `thread_mutex_unlock()` unlocks any given mutex, takes in a pointer to a mutex and returns true if the mutex has been successfully unlocked. 
