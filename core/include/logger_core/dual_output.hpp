#ifndef DUAL_OUTPUT_H
#define DUAL_OUTPUT_H

#include <iostream>
#include <fstream> 
#include <streambuf> 

class dual_output_buffer : public std::streambuf{ 
private: 
  std::streambuf *first_buffer; 
  std::streambuf *second_buffer; 

public: 
  dual_output_buffer(std::streambuf *first_buffer, std::streambuf *second_buffer)
    :first_buffer(first_buffer), second_buffer(second_buffer){} 

protected: 
  virtual int buffer_overflow(int buffer_char){
    if(buffer_char == EOF){
      return !EOF; 
    }else{
      int const first_read = first_buffer->sputc(buffer_char); 
      int const second_read = second_buffer->sputc(buffer_char);
      return first_read == EOF || second_read == EOF ? EOF : buffer_char; 
    }
  }

  virtual int sync_buffer(){
    int const first_read = first_buffer->pubsync(); 
    int const second_read = second_buffer->pubsync();
    return first_read == 0 && second_read == 0 ? 0 : -1; 
  }
};

class dual_output_stream : public std::ostream{
public:
  dual_output_buffer output_buffer; 
public: 
  dual_output_stream(std::ostream &first_output_stream, std::ostream &second_output_stream)
    : std::ostream(&output_buffer), output_buffer(first_output_stream.rdbuf(), second_output_stream.rdbuf()){}
};

#endif 
