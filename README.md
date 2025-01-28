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

## Code examples:

- Creating Tensors:
    ```c++
    /*The first parameter is the Tensor's rank,
    there must be dimensional parameters equal to the size of the tensor's rank,
    each dimension gets it's own size as well.*/
    
    Tensor::Tensor<float> tensor1(5,{5,5,5,5,5});
    Tensor::Tensor<int> tensor2(2,{10,10});
    
- Tensor operations:
    ```c++
    Tensor::Tensor<float> tensor1(3,{10,10,10});
    Tensor::Tensor<float> tensor2(3,{10,10,10});
    //Pass in previously created tensors through the TensorOp constructor to perform Tensor operations
    Tensor::TensorOps<float> ops1(tensor1);
    Tensor::TensorOps<float> ops2(tensor2);
    ops1.FillTensor();
    ops2.FillTensor();
    //Supports operator overloading and direct assigning to a new TensorOp
    TensorOps::TensorOps<float> ops3 = ops1 + ops2;
    TensorOps::TensorOps<float> ops4 = ops3 - ops2;
    TensorOps::TensorOps<float> ops5 = ops3 * ops4;
    //Zero's out all values in the Tensor
    ops5.zero()
    //Prints Tensor formatted according to it's dimensionality
    ops3.PrintTensor(); 

    //Output of a 3 dimensional 5x5x5 Tensor
    tensor([
  [
    [
      [5.8694, 9.0286, 5.1757, 3.3696, 2.9557],
      [2.3397, 4.0062, 2.6034, 8.2668, 5.7308],
      [7.0192, 1.3186, 1.6316, 2.9918, 6.1755],
      [5.2543, 3.8435, 8.0567, 1.7100, 6.8338],
      [9.3145, 9.3345, 1.8077, 1.8067, 7.3904]
    ],
    [
      [2.3397, 4.0062, 2.6034, 8.2668, 5.7308],
      [7.0192, 1.3186, 1.6316, 2.9918, 6.1755],
      [5.2543, 3.8435, 8.0567, 1.7100, 6.8338],
      [9.3145, 9.3345, 1.8077, 1.8067, 7.3904],
      [2.2128, 4.4945, 4.1580, 8.7090, 3.9576]
    ],
    [
      [7.0192, 1.3186, 1.6316, 2.9918, 6.1755],
      [5.2543, 3.8435, 8.0567, 1.7100, 6.8338],
      [9.3145, 9.3345, 1.8077, 1.8067, 7.3904],
      [2.2128, 4.4945, 4.1580, 8.7090, 3.9576],
      [4.4327, 5.9193, 9.3391, 6.4012, 3.5686]
    ],
    [
      [5.2543, 3.8435, 8.0567, 1.7100, 6.8338],
      [9.3145, 9.3345, 1.8077, 1.8067, 7.3904],
      [2.2128, 4.4945, 4.1580, 8.7090, 3.9576],
      [4.4327, 5.9193, 9.3391, 6.4012, 3.5686],
      [1.8384, 7.5324, 9.1052, 9.1920, 1.3079]
    ],
    [
      [9.3145, 9.3345, 1.8077, 1.8067, 7.3904],
      [2.2128, 4.4945, 4.1580, 8.7090, 3.9576],
      [4.4327, 5.9193, 9.3391, 6.4012, 3.5686],
      [1.8384, 7.5324, 9.1052, 9.1920, 1.3079],
      [9.1220, 2.5200, 1.1947, 6.8598, 7.3029]
    ]
  ]
  ])
- Creating Matrices:
    ```c++
    //Creates 2 5x5 Matrices
    mat::matrix<float> D(5,5); 
    mat::matrix<float> A(5,5);
    
- Matrix Operations:
    ```c++
    //Creates 2 5x5 Matrices
    mat::matrix<float> D(5,5); 
    mat::matrix<float> A(5,5);
    //Like Tensors we have to pass them through the MatOps constructor to gain access to all matrix operations
    MatOps::MatOps<float> ops1(A);
    MatOps::MatOps<float> ops2(D);
    ops1.fillMat();
    ops2.fillMat();
    //Supports operator overloading and direct assigning to a new MatOp
    mat::MatOps::MatOps<float> ops3 = ops1 + ops2;
    mat::MatOps::MatOps<float> ops4 = ops3 - ops2;
    mat::MatOps::MatOps<float> ops5 = ops3 * ops4;
    //Matrix Transpose
    ops1.TP();
    ops2.TP();
    //Zero's out the matrix
    ops1.zero();
    //Matrix Block, Retruns a user-defined sub-matrix from a larger one (Kernel)
    //i and j determine the starting points of the block, right and down respectively
    //p and q determine the step size of the block, right and down respectively
    //Block returns a regular Matrix or a direct MatOps object
    mat::matrix<float> C = ops1.block(size_t i, size_t j, size_t p, size_ q);        
    mat::MatOps::MatOps<float> ops6 = ops1.block(size_t i, size_t j, size_t p, size_ q);
    //Regular MatMul of two Matrices and MatOps is also available
    mat::matrix<float> E = mat::MatOps::MatOps<float>::matmul(A,D);
    mat::MatOps::MatOps<float> ops6 = mat::MatOps::MatOps<float>::matmul(ops1,ops2);
    //Scalar sum of a Matrix or MatOp is also available
    T sum = A.sum();
    T sum = ops1.sum();
    
    //Output of a 5x5 Matrix
    [
    0.135728, 0.360411, 0.280317, 0.766647, 0.380475,
    0.689877, 0.408091, 0.897573, 0.501442, 0.851538,
    0.386557, 0.207603, 0.419095, 0.45105, 0.685857,
    0.355872, 0.781755, 0.446186, 0.108611, 0.472237,
    0.95464, 0.185323, 0.506123, 0.951211, 0.517346,
    ]
    
- Creating Neural Networks:
    ```c++
    //Creates a net object
    Neural::nn net;
    //Adds a linear layer with 30 input nodes and 50 output nodes
    net.addLinear(30,50);
    //Adds a ReLU layer that takes in 50 input nodes
    net.addRelu(50);
    //Adds a Linear layer with 50 input nodes and 20 output nodes
    net.addLinear(50,25);
    net.addLeakyReLU(25);
    net.addLinear(25,10);
    net.addReLU(10);
    net.addSigmoid(10,1);
    
- Neural Networks Operations:
    ```c++
    //Regular Feed Forward function
    /*Depending on the Layer types and the number of layers,
    calling net.Forward() will initiate a callback sequence
    to each of the net's layers corresponding Forward() methods*/
    net.Forward();
    /*Depending on the Layer types and the number of layers,
    calling net.Backward() will initiate a callback sequence
    to each of the net's layers corresponding Backward() methods*/
    net.Backward();
    /*Depending on the Layer types and the number of layers,
    calling net.update(float eta) will initiate a callback sequence
    to each of the net's layers corresponding update() methods and update the learning rates*/
    net.update(float eta);

    //More Neural Net specific methods are being implemented... 
                       
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
