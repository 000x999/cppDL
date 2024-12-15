# cppML

An extensive and complete Machine Learning library written entirely in C++. 
<br><br>
**Features:**
- Tensors and Tensor operations. 
- Modularized forms of important Data Structures used in Machine learning.
- Multiple container types complete with operator overloading. 
- Extensive library of important functions used in Machine learning. 
- ***More to come ...*** 
<br><br>
## How It Works:
- ***A research paper and documentation will be written in due time.***
## **Notes**:
 ***STILL A WORK IN PROGRESS***
<br><br>
## **CURRENT FEATURES**:
- Currently supports Tensor operations with up to 312 Million parameters in a negligible amount of time.
- Currently supports Tensor transformations and with up to 312 Million parameters.
- Multiple activation functions have been implemented such as softmax, RELU, Sigmoid, SiLU, Softplus, SquarePlus and BinStep.
<br><br>
## ***CURRENTLY WORKING ON:***
- Custom memory alligned memory allocator to increase Tensor and Matrix operation and transformation speeds.
   - By alligning all data structures so they are contiguous, I estimate the ability to completely eliminate any loops/iterators used for Vector/Tensor operations. This will drastically increase performance as all data can be fetched from the next memory register in line. 
- Generalized Neural Network templates with the ability to be trained on any type of data set.
- Generalized Feed Forward and Back propagation methods.
- Image and Audio procressing functionalities.
- Tokenization.  
<br><br>
## Requirements
- C++17 or newer
- MinGW/GCC/G++
- GLM
<br><br>
## Usage instructions
- ***A comprehensive guide to use this library will be written in due time.***
<br><br>
### Building 
- Compile and run using the provided Makefile:
    ```bash
    - Make sure to have 'Make' installed, 'GCC'/'G++' as well as 'GLM'.  
    - Set your 'GLM' folder path in the CXXFLAGS section inside the Makefile. 
    - Simply run 'make' directly in the command line from within the 'cppML' folder.  
