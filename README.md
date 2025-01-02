# cppDL

An extensive and complete Deep Learning library written entirely in C++. 
<br><br>
## **CURRENT FEATURES**:
- Currently supports Tensor operations with up to 312 Million parameters in a negligible amount of time.
- Currently supports Tensor transformations and with up to 312 Million parameters.
- Multiple activation/attention functions have been implemented such as softmax, RELU, Sigmoid, SiLU, Softplus, SquarePlus and BinStep.
- Support for custom N-Sized Matrices as well as matrix addition, subtraction and multiplication through operator overloading. 
<br><br>
## ***CURRENTLY WORKING ON:***
- Custom memory alligned memory allocator to increase Tensor and Matrix operation and transformation speeds **HIGH PRIORITY**.
- Gradient descent and grad optim methods **HIGH PRIORITY**.
- Generalized Neural Network templates with the ability to be trained on any type of data set.
- Conv functions, Pooling functions, Attention functions/mechanisms (Some are already done), Loss functions, dropout functions, sparse functions and distance functions **HIGH PRIORITY** 
- Generalized Feed Forward and Back propagation methods.
- Image and Audio procressing functionalities.
- Tokenization.  
<br>

## How It Works:
- ***A research paper and documentation will be written in due time.***
## **Notes**:
 ***STILL A WORK IN PROGRESS***
<br><br>
## Requirements
- C++17 or newer
- MinGW/GCC/G++
<br><br>
## Usage instructions
- ***A comprehensive guide to use this library will be written in due time.***
<br><br>
### Building 
- Compile and run using the provided Makefile:
    ```bash
    - Make sure to have 'Make' installed and 'GCC'/'G++' 
    - Set your C++ STL 'Includes' folder path in the CXXFLAGS section inside the Makefile. 
    - Simply run 'make' directly in the command line from within the 'cppDL' folder.  
