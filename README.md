# cppDL
Extensive Deep Learning library written entirely in C++ STL without any external dependencies. 
<br><br>
## **CURRENT FEATURES**:
- Currently supports Tensor operations, transformations and dimensional reshaping with up to 312 Million parameters.
- Threaded Tiled Matrix Multiplications of 8192x8192 sized matrices using AVX256 instructions, in ~1.3s at 90GFLOP/s FP32 (CPU Bound).
- Multi-Threaded, Matrix Transpose of 16384x16384 sized matrices at ~0.8s at 2.55GB/s FP32 (CPU Bound).
- Custom Sized and Layered Neural Networks with attachable loss functions and optims. 
<br><br>
## ***CURRENTLY WORKING ON:***
- Optimized methods to increase Tensor and Matrix operations and transformation speeds (Currently being done through AVX256/512 and openMP, strictly CPU bound) **- MATMUL DONE ✅ - TENSOR MUL ⏳**
- Custom GPU Kernel backend and Kernel execution pipeline using PTX, SPIR-V and SIMT **- IN PROGRESS ⏳**
- Gradient descent and grad optim methods **- HIGH PRIORITY**.
- Generalized Neural Network template **- DONE ✅** 
- Conv functions, Pooling functions, Attention functions/mechanisms (Some are already done), Loss functions, dropout functions, sparse functions and distance functions **HIGH PRIORITY** 
- Generalized Feed Forward and Back propagation methods **- DONE ✅**
- Image and Audio procressing functionalities.
- Tokenization.
- ***NOTE 1:*** **Currently refactoring and switching to a Hybrid DOD approach as opposed to a typical OOP approach, I found through some separate testing that this greatly increased performance AND reduced overall memory consumption.**
             **This switch will make writing code in the library slighty more verbose/elaborate. A Hybrid implementation angle is the best move at the moment as it somewhat caps how verbose the code becomes. Obviously this is still in early stages of development so more changes might surface later down the line.** ***!!!!***
- ***NOTE 2:*** **All the code/structuring for the library that is currently available is NOT final, I am mainly looking to get a working skeleton for the library to allow extensive testing and quick edits to the code. Once the library is in a semi-decent state I will go back and smooth everything out, optimize, implement more robust error checking as well as**
             **Macros, pre-processor implementations for SIMD, multi-platform support/specific operations and architectures, Compiler specific directives and so on.** ***!!!!***

## Benchmarks:
- MatMul Benchmark:
    ```c++
    void MatMulBenchmark(float A, int blocksize){
       double totalOps = 2.0 * double(A) * double(A) * double(A);
       double gflopFactor = 1.0e-9;
       std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl; 
       mat::matrix<float> mat1(A, A);
       mat::matrix<float> mat2(A, A); 
       mat::MatOps<float> op1(mat1); 
       mat::MatOps<float> op2(mat2);
       op1.fillMat(); 
       op2.fillMat(); 
       op1.setBlockSize(blocksize); 
       auto start = nanos(); 
       mat::MatOps<float> op3 = op1 * op2; 
       auto end = nanos(); 
       double optTime = (end - start) * 1e-9;
       double optGflops = (totalOps * gflopFactor) / optTime;
       std::cout << "AVX MatMul: " << optTime
       << "s, GFLOP/S = " << optGflops << "\n";
    }
    
    int main(){
     //First argument is the size of the matrix (Multiple of 8)
     //Second argument is the block size for the MatMul
     MatMulBenchmark(1024, 8);
    }

    //==========Benchmark output==========:
    2.14748 GFLOP
    //Cpu specific thread activation info will appear here...
    Thread 9 out of 16
    Thread 4 out of 16
    Thread 1 out of 16
    Thread 11 out of 16
    Thread 12 out of 16
    Thread 14 out of 16
    Thread 8 out of 16
           ...
    AVX MatMul: 0.0110571s, GFLOP/S = 194.217

- Mattrix Transpose:
    ```c++
    void TransposeBenchmark(float A){
       double totalOps =  double(A) * double(A);
       double memfactor = 2.0 * A *  A * sizeof(float);
       double memfactorgb = memfactor / (1024.0 * 1024.0 * 1024.0); 
       std::cout<< totalOps * 1e-6<< " KB" << std::endl; 
       mat::matrix<float> mat1(A, A);
       mat::MatOps<float> op1(mat1); 
       op1.fillMat();
       auto start = nanos();
       op1.TP();
       auto end = nanos(); 
       double optTime = (end - start) * 1e-9;
       double optmem =  memfactorgb / optTime;
       std::cout << "Transpose: " << optTime
          << "s, GB/S = " << optmem << "\n";
    }
    
    int main(){
     //First argument is the size of the matrix 
     TransposeBenchmark(16384);
    }
    
    //==========Benchmark output==========:
    268.435 KB
    //Cpu specific thread activation info will appear here...
    Thread 9 out of 16
    Thread 4 out of 16
    Thread 1 out of 16
    Thread 11 out of 16
    Thread 12 out of 16
    Thread 14 out of 16
    Thread 8 out of 16
           ...
    Transpose: 0.842949s, GB/S = 2.37262
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
    Tensor::TensorOps<float> ops3 = ops1 + ops2;
    Tensor::TensorOps<float> ops4 = ops3 - ops2;
    Tensor::TensorOps<float> ops5 = ops3 * ops4;
    //Zero's out all values in the Tensor
    ops5.zero()
    //Prints Tensor formatted according to it's dimensionality
    ops3.PrintTensor(); 

    //Output of a 3 dimensional 5x5x5 Tensor
    tensor(
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
    mat::MatOps<float> ops1(A);
    mat::MatOps<float> ops2(D);
    ops1.fillMat();
    ops2.fillMat();
    //Supports operator overloading and direct assigning to a new MatOp
    mat::MatOps<float> ops3 = ops1 + ops2;
    mat::MatOps<float> ops4 = ops3 - ops2;
    mat::MatOps<float> ops5 = ops3 * ops4;
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
    mat::MatOps<float> ops6 = ops1.block(size_t i, size_t j, size_t p, size_ q);
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
  Neural::nn net;
  net.addLinear(3,2); 
  net.addRelu(2); 
  net.addLinear(2,1);
  //Attaches a Loss function to the neural network
  //using make_unique ensures the neural networks ownership of the loss function
  net.addLoss(std::make_unique<Neural::MSEloss>());
  
  std::vector<float> inputVals = {15.6f, -25.1f, 33.5f}; 
  std::vector<float> targetVals = {1.0f};

  float eta = 0.1f; 
  for(size_t epoch = 0; epoch < 100; ++epoch){
    //Returns a vector from the Feed forward callback sequence
    auto out = net.Forward(inputVals);
    //Computes the loss based off of the target value vector
    float loss = net.getLoss(targetVals); 
    std::cout << "Epoch: " << epoch << "Loss: " << loss << " | Output = " << out[0] << " | Target = " << targetVals[0] << std::endl;
    //Computes the loss gradient
    auto derivOut  = net.getGrad(targetVals);
    //Initiates the back propagation callback sequence
    net.Backwards(derivOut);
    //Initiates the callback sequence to update the learning rate
    net.update(eta);
  }

  //Example output of this Neural network is as follows
   Epoch: 0 | Loss: 139.941 | Output[1] = 17.7297 | Target = 1
   Epoch: 1 | Loss: 4.80543 | Output[1] = -2.10014 | Target = 1
   Epoch: 2 | Loss: 3.8924 | Output[1] = -1.79012 | Target = 1
   Epoch: 3 | Loss: 3.15284 | Output[1] = -1.51111 | Target = 1
   Epoch: 4 | Loss: 2.5538 | Output[1] = -1.26 | Target = 1
   Epoch: 5 | Loss: 2.06858 | Output[1] = -1.034 | Target = 1
   Epoch: 6 | Loss: 1.67555 | Output[1] = -0.830601 | Target = 1
   Epoch: 7 | Loss: 1.35719 | Output[1] = -0.64754 | Target = 1
   Epoch: 8 | Loss: 1.09933 | Output[1] = -0.482786 | Target = 1
   Epoch: 9 | Loss: 0.890455 | Output[1] = -0.334508 | Target = 1
   ... A few dozen epochs later ...
   Epoch: 91 | Loss: 2.78983e-08 | Output[1] = 0.999764 | Target = 1
   Epoch: 92 | Loss: 2.26015e-08 | Output[1] = 0.999787 | Target = 1
   Epoch: 93 | Loss: 1.83038e-08 | Output[1] = 0.999809 | Target = 1
   Epoch: 94 | Loss: 1.4826e-08 | Output[1] = 0.999828 | Target = 1
   Epoch: 95 | Loss: 1.20082e-08 | Output[1] = 0.999845 | Target = 1
   Epoch: 96 | Loss: 9.72662e-09 | Output[1] = 0.999861 | Target = 1
   Epoch: 97 | Loss: 7.87856e-09 | Output[1] = 0.999874 | Target = 1
   Epoch: 98 | Loss: 6.37894e-09 | Output[1] = 0.999887 | Target = 1 
   Epoch: 99 | Loss: 5.16997e-09 | Output[1] = 0.999898 | Target = 1
                            
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
