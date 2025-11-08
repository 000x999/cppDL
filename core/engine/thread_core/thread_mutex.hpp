#ifndef THREADMUTEX_HPP
#define THREADMUTEX_HPP
#include "../../defines.h"

class thread_mutex{
public:
  void *internal_mutex_data; 
  virtual ~thread_mutex() = default; 
  CPPDL_API virtual bool thread_mutex_create (); 
  CPPDL_API virtual void thread_mutex_destroy();
  CPPDL_API virtual bool thread_mutex_lock   (); 
  CPPDL_API virtual bool thread_mutex_unlock (); 
};
#endif
