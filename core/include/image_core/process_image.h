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
struct IMGHead {
    //Declare all u16's before u32's to reduce bit padding and ensure alignment 
    //this also reduces mem size for each instance of this struct
    uint16_t fileType{0x4D42};
    uint16_t reserved1{0};   
    uint16_t reserved2{0};  
    uint32_t fileSize{0}; 
    uint32_t dataOffset{0};
};   
struct IMGInfo {
  //Declare all u16's before u32s again for the same reasons 
  //u32's come before i32's again to elemina te bit padding reqs
  uint16_t planes{ 1 };
  uint16_t bpp{ 0 };
  uint32_t compression{ 0 }; 
  uint32_t imgSize{ 0 };   
  uint32_t size{ 0 };
  uint32_t activeCols{ 0 };     
  uint32_t colorsImp{ 0 };
  int32_t width{ 0 };
  int32_t height{ 0 };
  int32_t xpp{ 0 };
  int32_t ypp{ 0 };
};

struct IMGCol {
  //No padding check needed, all are u32's
  uint32_t redM{0x00ff0000}; 
  uint32_t greenM{0x0000ff00}; 
  uint32_t blueM{0x000000ff};
  uint32_t alphaM{0xff00000}; 
  uint32_t colorSpace{0x73524742}; 
  uint32_t unused[16]{0}; 
};

struct BMP{
  IMGHead fileH; 
  IMGInfo bmpInfo; 
  IMGCol bmpCol; 
  std::vector<uint8_t> data; 

  BMP(const char* fname){
   
  }
  
  void read(const char* fname){
    std::ifstream inp{fname, std::ios_base::binary};
    if(inp){
      inp.read((char*)&fileH, sizeof(fileH));
      if(fileH.fileType != 0x4D42){
        throw std::runtime_error("Error!, wrong file format"); 
      }
      inp.read((char*)&bmpInfo, sizeof(bmpInfo)); 
      if(bmpInfo.bpp == 32){
        if(bmpInfo.size >= (sizeof(IMGInfo) + sizeof(IMGCol)))
          inp.read((char*)&bmpCol, sizeof(bmpCol)); 
      }else{
        std::cerr<<"Warning! the file \"" << fname << "\" does not seem to contain the bit mask info\n"; 
        throw std::runtime_error("Error! wrong file format"); 
      }
    }
    //jump to the pixel data location 
    inp.seekg(fileH.dataOffset, inp.beg);
    //Adjust header fields for output
    if(bmpInfo.bpp == 32){
      bmpInfo.size = sizeof(IMGInfo) + sizeof(IMGCol);
      fileH.dataOffset = sizeof(IMGHead) + sizeof(IMGInfo) + sizeof(IMGCol); 
    }else{
      bmpInfo.size = sizeof(IMGInfo); 
      fileH.dataOffset = sizeof(IMGHead) + sizeof(IMGInfo);
    }
    fileH.fileSize = fileH.dataOffset; 
    if(bmpInfo.height < 0){
      throw std::runtime_error("Only BMP images with positive height are valid");
    }
    data.resize(bmpInfo.width * bmpInfo.height * bmpInfo.bpp / 8);
    //Check to see if we need padding or not 
    if(bmpInfo.width % 4 == 0){
      inp.read((char*)data.data(), data.size()); 
      fileH.fileSize += data.size();
    }else{
      rowStride = bmpInfo.width * bmpInfo.bpp / 8;
      uint32_t newStride = AlignStride(4);
      std::vector<uint8_t> paddingRow(newStride - rowStride); 
      for(size_t y = 0; y < bmpInfo.height; ++y){
        inp.read((char*)(data.data() + rowStride * y), rowStride);
        inp.read((char*)paddingRow.data(), paddingRow.size()); 
      }
      fileH.fileSize += data.size() + bmpInfo.height*paddingRow.size();
    } 
  }

  BMP(int32_t width, int32_t height, bool alpha = true){
    if(width <= 0 || height <=0){
      throw std::runtime_error("width and height must be positive");
    }
    bmpInfo.width = width; 
    bmpInfo.height = height; 
    if(alpha){
      bmpInfo.size = sizeof(IMGInfo) + sizeof(IMGCol); 
      fileH.dataOffset = sizeof(IMGHead) + sizeof(IMGInfo) + sizeof(IMGCol); 
      bmpInfo.bpp = 32;
      bmpInfo.compression = 3; 
      rowStride = width * 4;
      data.resize(rowStride * height); 
      fileH.fileSize = fileH.dataOffset + data.size();
    }else{
      bmpInfo.size = sizeof(IMGInfo); 
      fileH.dataOffset = sizeof(IMGHead) + sizeof(IMGInfo);
      bmpInfo.bpp = 24;
      bmpInfo.compression = 0; 
      rowStride = width * 3;
      data.resize(rowStride * height);
      uint32_t newStride = AlignStride(4);
      fileH.fileSize = fileH.dataOffset + data.size() + bmpInfo.height * (newStride - rowStride);
    }
  }
  
  void fill(uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, uint8_t B, uint8_t G, uint8_t R, uint8_t A){
    if(x0 + w > (uint32_t)bmpInfo.width||y0+h > (uint32_t)bmpInfo.height){
      throw std::runtime_error("The region doesn't fit in the image"); 
    }
    uint32_t channels = bmpInfo.bpp / 8; 
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
      if(bmpInfo.bpp == 32){
        WriteHeadData(of);
      }else if(bmpInfo.bpp == 24){
        if(bmpInfo.width % 4 == 0){
          WriteHeadData(of);
        }else{
          uint32_t newStride = AlignStride(4);
          std::vector<uint8_t> paddingRow(newStride - rowStride); 
          WriteHead(of);
          for(size_t y = 0; y < bmpInfo.height; ++y){
            of.write((const char*)(data.data() + rowStride*y), rowStride);
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
 uint32_t rowStride{ 0 };

  void WriteHead(std::ofstream &of){
    of.write((const char*) &fileH, sizeof(fileH)); 
    of.write((const char*)&bmpInfo, sizeof(bmpInfo)); 
    if(bmpInfo.bpp==32){
      of.write((const char*)&bmpCol, sizeof(bmpCol)); 
    }
  }

 void WriteHeadData(std::ofstream &of){
    WriteHead(of); 
    of.write((const char*) data.data(), data.size());
  }
  
 uint32_t AlignStride(uint32_t restStride){
    uint32_t newStride = rowStride; 
    while(newStride % restStride != 0){
      newStride++; 
    }
    return newStride; 
  } 

  void HeadColor(IMGCol &bmpCol){
    IMGCol expectedCol; 
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
