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
- ***NOTE 1:*** **Currently refactoring and switching to a Hybrid DOD approach as opposed to a typical OOP approach, I found through some separate testing that this greatly increased performance AND reduced overall memory consumption.**
             **This switch will make writing code in the library slighty more verbose/elaborate. A Hybrid implementation angle is the best move at the moment as it somewhat caps how verbose the code becomes. Obviously this is still in early stages of development so more changes might surface later down the line.** ***!!!!***
- ***NOTE 2:*** **All the code/structuring for the library that is currently available is NOT final, I am mainly looking to get a working skeleton for the library to allow extensive testing and quick edits to the code. Once the library is in a semi-decent state I will go back and smooth everything out, optimize, implement more robust error checking as well as**
             **Macros, pre-processor implementations for SIMD, multi-platform support/specific operations and architectures, Compiler specific directives and so on.** ***!!!!***
            
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
