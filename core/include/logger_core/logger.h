#ifndef LOGGER_H
#define LOGGER_H
#include "../../defines.h"

#define CPPDL_LOG_WARN_ENABLED  1
#define CPPDL_LOG_INFO_ENABLED  1
#define CPPDL_LOG_DEBUG_ENABLED 1

typedef enum cppdl_log_level{
  CPPDL_LOG_LEVEL_FATAL = 0, 
  CPPDL_LOG_LEVEL_ERROR = 1, 
  CPPDL_LOG_LEVEL_WARN  = 2, 
  CPPDL_LOG_LEVEL_INFO  = 3, 
  CPPDL_LOG_LEVEL_TRACE = 4, 
  CPPDL_LOG_LEVEL_DEBUG = 5
}cppdl_log_level;

bool init_logging(); 
bool suspend_logging(); 

template <typename... Args>
inline CPPDL_API void log_output(cppdl_log_level log_level, Args&&... args){
  static constexpr const char *log_level_strings[] = {"\033[31m[FATAL]", "\033[31m[ERROR]", "\033[33m[WARN]","\033[32m[INFO]", "[DEBUG]"};
  bool is_error = log_level < 2; 
  std::ostringstream log_string_stream; 
  log_string_stream << log_level_strings[log_level] << " "; 
  (void)(log_string_stream << ... << std::forward<Args>(args)); 
  auto log_string_buffer = log_string_stream.str(); 
  FILE *log_message_out = is_error ? stderr : stdout;
  std::fwrite(log_string_buffer.c_str(), 1, log_string_buffer.size(), log_message_out); 
}
template CPPDL_API void log_output<>(cppdl_log_level);
#ifndef CPPDL_FATAL
  #define CPPDL_FATAL(...)\
  do{\
    log_output(CPPDL_LOG_LEVEL_FATAL, __VA_ARGS__);\
  }while(0)
#endif

#ifndef CPPDL_ERROR
  #define CPPDL_ERROR(...)\
  do{\
    log_output(CPPDL_LOG_LEVEL_ERROR, __VA_ARGS__);\
  }while(0)
#endif 

#if CPPDL_lOG_WARN_ENABLED == 1
  #define CPPDL_WARN(...)\
  do{\
    log_output(CPPDL_LOG_LEVEL_WARN, __VA_ARGS__);\
  }while(0)
#else 
  #define CPPDL_WARN(...)
#endif 

#if CPPDL_LOG_INFO_ENABLED == 1
  #define CPPDL_INFO(...)\
  do{\
    log_output(CPPDL_LOG_LEVEL_INFO, __VA_ARGS__);\
  }while(0)
#else 
  #define CPPDL_INFO(...)
#endif 

#if CPPDL_LOG_TRACE_ENABLED == 1
  #define CPPDL_TRACE(...)\
  do{\
    log_output(CPPDL_LOG_LEVEL_TRACE, __VA_ARGS__);\
  }while(0)
#else 
  #define CPPDL_TRACE(...)
#endif 

#if CPPDL_LOG_DEBUG_ENABLED == 1
  #define CPPDL_DEBUG(...)\
  do{\
    log_output(CPPDL_LOG_LEVEL_DEBUG, __VA_ARGS__);\
  }while(0)
#else 
  #define CPPDL_DEBUG(...)
#endif
#endif 
