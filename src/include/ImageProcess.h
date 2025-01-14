#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#pragma once
#include <iostream> 
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <ctime> 
#include <cstdlib> 
#include <cstdint>

#pragma pack(push, 4)
struct BMPFileHeader {
    //Declare all u16's before u32's to reduce bit padding and ensure alignment 
    //this also reduces mem size for each instance of this struct
    uint16_t file_type{0x4D42};
    uint16_t reserved1{0};   
    uint16_t reserved2{0};  
    uint32_t file_size{0}; 
    uint32_t offset_data{0};
};   
struct BMPInfoHeader {
  //Declare all u16's before u32s again for the same reasons 
  //u32's come before i32's again to elemina te bit padding reqs
  uint16_t planes{ 1 };
  uint16_t bit_count{ 0 };
  uint32_t compression{ 0 }; 
  uint32_t size_image{ 0 };   
  uint32_t size{ 0 };
  uint32_t colors_used{ 0 };     
  uint32_t colors_important{ 0 };
  int32_t width{ 0 };
  int32_t height{ 0 };
  int32_t x_pixels_per_meter{ 0 };
  int32_t y_pixels_per_meter{ 0 };
};

struct BMPColorHeader {
  //No padding check needed, all are u32's
  uint32_t redM{0x00ff0000}; 
  uint32_t greenM{0x0000ff00}; 
  uint32_t blueM{0x000000ff};
  uint32_t alphaM{0xff00000}; 
  uint32_t colorSpace{0x73524742}; 
  uint32_t unused[16]{0}; 
};

struct BMP{
  BMPFileHeader fileH; 
  BMPInfoHeader bmpInfo; 
  BMPColorHeader bmpCol; 
  std::vector<uint8_t> data; 

  BMP(const char* fname){
   
  }
  
  void read(const char* fname){
    std::ifstream inp{fname, std::ios_base::binary};
    if(inp){
      inp.read((char*)&fileH, sizeof(fileH));
      if(fileH.file_type != 0x4D42){
        throw std::runtime_error("Error!, wrong file format"); 
      }
      inp.read((char*)&bmpInfo, sizeof(bmpInfo)); 
      if(bmpInfo.bit_count == 32){
        if(bmpInfo.size >= (sizeof(BMPInfoHeader) + sizeof(BMPColorHeader)))
          inp.read((char*)&bmpCol, sizeof(bmpCol)); 
      }else{
        std::cerr<<"Warning! the file \"" << fname << "\" does not seem to contain the bit mask info\n"; 
        throw std::runtime_error("Error! wrong file format"); 
      }
    }
    //jump to the pixel data location 
    inp.seekg(fileH.offset_data, inp.beg);
    //Adjust header fields for output
    if(bmpInfo.bit_count == 32){
      bmpInfo.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
      fileH.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader); 
    }else{
      bmpInfo.size = sizeof(BMPInfoHeader); 
      fileH.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    }
    fileH.file_size = fileH.offset_data; 
    if(bmpInfo.height < 0){
      throw std::runtime_error("Only BMP images with positive height are valid");
    }
    data.resize(bmpInfo.width * bmpInfo.height * bmpInfo.bit_count / 8);
    //Check to see if we need padding or not 
    if(bmpInfo.width % 4 == 0){
      inp.read((char*)data.data(), data.size()); 
      fileH.file_size += data.size();
    }else{
      row_stride = bmpInfo.width * bmpInfo.bit_count / 8;
      uint32_t newStride = make_stride_aligned(4);
      std::vector<uint8_t> paddingRow(newStride - row_stride); 
      for(size_t y = 0; y < bmpInfo.height; ++y){
        inp.read((char*)(data.data() + row_stride * y), row_stride);
        inp.read((char*)paddingRow.data(), paddingRow.size()); 
      }
      fileH.file_size += data.size() + bmpInfo.height*paddingRow.size();
    } 
  }

  BMP(int32_t width, int32_t height, bool alpha = true){
    if(width <= 0 || height <=0){
      throw std::runtime_error("width and height must be positive");
    }
    bmpInfo.width = width; 
    bmpInfo.height = height; 
    if(alpha){
      bmpInfo.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader); 
      fileH.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader); 
      bmpInfo.bit_count = 32;
      bmpInfo.compression = 3; 
      row_stride = width * 4;
      data.resize(row_stride * height); 
      fileH.file_size = fileH.offset_data + data.size();
    }else{
      bmpInfo.size = sizeof(BMPInfoHeader); 
      fileH.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
      bmpInfo.bit_count = 24;
      bmpInfo.compression = 0; 
      row_stride = width * 3;
      data.resize(row_stride * height);
      uint32_t newStride = make_stride_aligned(4);
      fileH.file_size = fileH.offset_data + data.size() + bmpInfo.height * (newStride - row_stride);
    }
  }
  
  void fill(uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, uint8_t B, uint8_t G, uint8_t R, uint8_t A){
    if(x0 + w > (uint32_t)bmpInfo.width||y0+h > (uint32_t)bmpInfo.height){
      throw std::runtime_error("The region doesn't fit in the image"); 
    }
    uint32_t channels = bmpInfo.bit_count / 8; 
    for(uint32_t y = y0; y < y0 + h; ++y){
      for(uint32_t x = x0; x < x0 + w; ++x){
        data[channels * (y * bmpInfo.width + x) + 0] = B; 
        data[channels * (y * bmpInfo.width + x) + 1] = G;
        data[channels * (y * bmpInfo.width + x) + 2] = R;
        if(channels == 4){
          data[channels * (y * bmpInfo.width + x) + 3] = A;
        }
      }
    }
  }

  void write(const char* fname){
    std::ofstream of{fname, std::ios_base::binary}; 
    if(of){
      if(bmpInfo.bit_count == 32){
        write_headers_and_data(of);
      }else if(bmpInfo.bit_count == 24){
        if(bmpInfo.width % 4 == 0){
          write_headers_and_data(of);
        }else{
          uint32_t newStride = make_stride_aligned(4);
          std::vector<uint8_t> paddingRow(newStride - row_stride); 
          write_headers(of);
          for(size_t y = 0; y < bmpInfo.height; ++y){
            of.write((const char*)(data.data() + row_stride*y), row_stride);
            of.write((const char*)paddingRow.data(), paddingRow.size());
          }
        }
      }
      else{
        throw std::runtime_error("BPP OOR"); 
      }
    }
    else{
      throw std::runtime_error("Unable to open file"); 
    }
  }

private:
 uint32_t row_stride{ 0 };

  void write_headers(std::ofstream &of){
    of.write((const char*) &fileH, sizeof(fileH)); 
    of.write((const char*)&bmpInfo, sizeof(bmpInfo)); 
    if(bmpInfo.bit_count==32){
      of.write((const char*)&bmpCol, sizeof(bmpCol)); 
    }
  }

 void write_headers_and_data(std::ofstream &of){
    write_headers(of); 
    of.write((const char*) data.data(), data.size());
  }
  
 uint32_t make_stride_aligned(uint32_t align_stride){
    uint32_t newStride = row_stride; 
    while(newStride % align_stride != 0){
      newStride++; 
    }
    return newStride; 
  } 

  void check_color_header(BMPColorHeader &bmpCol){
    BMPColorHeader expectedCol; 
    if(expectedCol.redM != bmpCol.redM || expectedCol.blueM != bmpCol.blueM ||
     expectedCol.greenM != bmpCol.greenM ||expectedCol.alphaM != bmpCol.alphaM){
      throw std::runtime_error("Unexpected color format");
    }
    if(expectedCol.colorSpace != bmpCol.colorSpace){
      throw std::runtime_error("Unexpected color space"); 
    }
  }
};
#pragma pack(pop)
#endif 
