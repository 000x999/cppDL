# cppDL
- Extensive Deep Learning/Machine Learning library written entirely in C++ STL and VULKAN. 
- Uses my custom BLAS/Encryption/Compression library CRUSHBLAS.
<br><br>
## **CURRENT FEATURES**:
- Custom Sized and Layered Neural Networks with attachable loss functions and optims.
- Text tokenization through Byte pair encoding, automatic grammar generation through token merge rules, vocabulary saving/loading, encoding and decoding of text.
- Generalized Matrix API and Matrix operations container using a single contiguous array structure. 
- Threaded Tiled Matrix Multiplications of matrices using AVX256 instructions, speed will depend on the machine, but one can expect 4096x4096 matrices in ~0.51s at 265 GFLOP/s FP32 (CPU Bound).
- Multi-Threaded and Tiled, Matrix Transpose of matrices using AVX256 instructions , speed will depend on the machine, but one can expect 16384x16384 matrices in ~0.889s at 2.25 GB/s FP32 (CPU Bound).
- Level3 GEMM kernel using the custom Matrix API as well as AVX256 instruction acceleration.
- Generalized Tensor API and Tensor operations container through the custom Matrix API.
- Contracted TensorMul using the level3 GEMM kernel with AVX256 acceleration, speed will depend on the machine and the Tensor structure, but one can expect a Contracted TensorMul with,  ```Tensor(1,15,4096)```, that is, 1 batch, 15 slices of 4096x4096 matrices to be computed in ~8.1893s at 251.741 GFLOP/s FP32 (CPU Bound).
- Batched TensorMul using the level3 GEMM kernel with AVX256 acceleration, speed will depend on the machine and the Tensor structure, but one can expect a Batched TensorMul with,  ```Tensor(1,15,4096)```, that is, 1 batch, 15 slices of 4096x4096 matrices to be computed in ~8.1893s at 237.869 GFLOP/s FP32 (CPU Bound).
<br><br>
## ***CURRENTLY WORKING ON:***
- Optimized methods to increase Tensor and Matrix operations and transformation speeds (Currently being done through AVX256/512 and openMP, strictly CPU bound) **- MATMUL DONE ✅ - TENSOR MUL ⏳**
- ~Custom GPU Kernel backend and Kernel execution pipeline using PTX, SPIR-V and SIMT **- IN PROGRESS ⏳**~
- I'm tabling the custom GPU kernel, Drivers and Execution pipeline for now... The furthest I got with a custom driver was detecting and talking to the GPU (On windows) but anything beyond that is outside my current knowledge unfortunately. I'll be switching to a custom VULKAN Computer Shader backend for all GPU operation offloading, I'll be using VULKAN since it's readily available on any GPU and doesn't rely on maker specific BLAS libraries like CUDA or ROCm.
- Gradient descent and grad optim methods **- HIGH PRIORITY**.
- Generalized Neural Network template **- DONE ✅** 
- Conv functions, Pooling functions, Attention functions/mechanisms (Some are already done), Loss functions, dropout functions, sparse functions and distance functions **HIGH PRIORITY** 
- Generalized Feed Forward and Back propagation methods **- DONE ✅**
- Image and Audio procressing functionalities.
- Tokenization **- DONE ✅ -BPE Tokenizer, Saving vocabularies, Loading vocabularies** **- IN PROGRESS: -Text modeling, -Text generation and prediction, -SIMD and Multi-Threading optimizations still need to be done⏳** 
- ***NOTE 1:*** **Currently refactoring and switching to a Hybrid DOD approach as opposed to a typical OOP approach, I found through some separate testing that this greatly increased performance AND reduced overall memory consumption.**
             **This switch will make writing code in the library slighty more verbose/elaborate. A Hybrid implementation angle is the best move at the moment as it somewhat caps how verbose the code becomes. Obviously this is still in early stages of development so more changes might surface later down the line.** ***!!!!***
- ***NOTE 2:*** **All the code/structuring for the library that is currently available is NOT final, I am mainly looking to get a working skeleton for the library to allow extensive testing and quick edits to the code. Once the library is in a semi-decent state I will go back and smooth everything out, optimize, implement more robust error checking as well as**
             **Macros, pre-processor implementations for SIMD, multi-platform support/specific operations and architectures, Compiler specific directives and so on.** ***!!!!***

### BENCHMARKS
- Contracted TensorMul Benchmark:
    ```c++
    void contracted_tensor_mul_benchmark(size_t batches, size_t slices, size_t matrix_size){
      tens::tensor tensor_a(dims,rank,matrix_size);
      tens::tensor tensor_b(dims,rank,matrix_size);
      tens::tensor_ops tensor_op_a(tensor_a);
      tens::tensor_ops tensor_op_b(tensor_b); 
      tens::tensor_ops::fill_tensor(tensor_op_a);
      tens::tensor_ops::fill_tensor(tensor_op_b);
      std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;
      std::cout << "Tensor batches: " << tensor_a.m_batches << std::endl; 
      std::cout << "Tensor slices: " << tensor_a.m_slices << std::endl;
      double totalOps = tensor_a.m_slices * (2 * double(matrix_size) * double(matrix_size) * double(matrix_size));
      double gflopFactor = 1.0e-9;
      std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl;
      auto start = nanos();
      mat::mat_ops C_mat = tens::tensor_ops::contract_tensor_mul(tensor_op_a, tensor_op_b);
      auto end = nanos();
      double optTime = (end - start) * 1e-9;
      double optGflops = (totalOps * gflopFactor) / optTime;
      std::cout << "AVX CONTRACTED TENSOR MUL: " << optTime
                << "s, GFLOP/S = " << optGflops << "\n";
    }

    int main(){
        //First argument is the total number of tensor batches
        //The second argument is the total number of tensor slices
        //The third argument is the size of the matrices in each tensor slice
        /*Matrices should be multiples of 8 and MINIMUM 256x256 to make use of AVX256 optimizations
        Otherwise any NxN or NxM sized matrix will work just fine but won't be accelerated through AVX256*/
        contracted_tensor_mul_benchmark(1,15,4096)
    }

    //==========Benchmark output==========:
    Matrix size: 4096x4096
    Tensor batches: 1
    Tensor slices: 15
    2061.58 GFLOP
    AVX CONTRACTED TENSOR MUL: 8.1893s, GFLOP/S = 251.741

- Batched TensorMul Benchmark:
    ```c++
    void batched_tensor_mul_benchmark(size_t batches, size_t slices, size_t matrix_size){
      tens::tensor tensor_a(dims,rank,matrix_size);
      tens::tensor tensor_b(dims,rank,matrix_size);
      tens::tensor_ops tensor_op_a(tensor_a);
      tens::tensor_ops tensor_op_b(tensor_b); 
      tens::tensor_ops::fill_tensor(tensor_op_a);
      tens::tensor_ops::fill_tensor(tensor_op_b);
      std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;
      std::cout << "Tensor batches: " << tensor_a.m_batches << std::endl; 
      std::cout << "Tensor slices: " << tensor_a.m_slices << std::endl;
      double totalOps = tensor_a.m_slices * (2 * double(matrix_size) * double(matrix_size) * double(matrix_size));
      double gflopFactor = 1.0e-9;
      std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl;
      auto start = nanos();
      tens::tensor_ops tensor_c = tens::tensor_ops::batch_tensor_mul(tensor_op_a, tensor_op_b);
      auto end = nanos();
      double optTime = (end - start) * 1e-9;
      double optGflops = (totalOps * gflopFactor) / optTime;
      std::cout << "AVX BATCHED TENSOR MUL: " << optTime
                << "s, GFLOP/S = " << optGflops << "\n";
    }
    
    int main(){
        //First argument is the total number of tensor batches
        //The second argument is the total number of tensor slices
        //The third argument is the size of the matrices in each tensor slice
        /*Matrices should be multiples of 8 and MINIMUM 256x256 to make use of AVX256 optimizations
        Otherwise any NxN or NxM sized matrix will work just fine but won't be accelerated through AVX256*/
        batched_tensor_mul_benchmark(1,15,4096)
    }

    //==========Benchmark output==========:
    Matrix size: 4096x4096
    Tensor batches: 1
    Tensor slices: 15
    2061.58 GFLOP
    AVX BATCHED TENSOR MUL: 8.66689s, GFLOP/S = 237.869
    
- MatMul Benchmark:
    ```c++
   void MatMulBenchmark(float A){
      double totalOps = 2.0 * double(A) * double(A) * double(A);
      double gflopFactor = 1.0e-9;
      std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl; 
      mat::matrix mat1(A, A);
      mat::matrix mat2(A, A); 
      mat::mat_ops op1(mat1); 
      mat::mat_ops op2(mat2);
      op1.fill_mat();
      op2.fill_mat(); 
      auto start = nanos(); 
      mat::mat_ops op3 = mat::matops::mat_mul(op1,op2); 
      auto end = nanos(); 
      double optTime = (end - start) * 1e-9;
      double optGflops = (totalOps * gflopFactor) / optTime;
      std::cout << "AVX MatMul: " << optTime
                << "s, GFLOP/S = " << optGflops << "\n";
    }

    int main(){
       //First argument is the size of the matrix (Multiple of 8)
       MatMulBenchmark(4096);
    }

    //==========Benchmark output==========:
    137.439 GFLOP
    DEBUG: AVX_MATMUL_STARTED
    AVX MatMul: 0.517075s, GFLOP/S = 265.801


- Transpose Benchmark:
    ```c++
   void TransposeBenchmark(float A){
      double totalOps =  double(A) * double(A);
      double memfactor = 2.0 * A *  A * sizeof(float);
      double memfactorgb = memfactor / (1024.0 * 1024.0 * 1024.0); 
      std::cout<< totalOps * 1e-6<< " KB" << std::endl; 
      mat::matrix mat1(A, A);
      mat::mat_ops op1(mat1); 
      op1.fill_mat();
      auto start = nanos();
      op1 = mat::mat_ops::transpose_matrix(op1);
      auto end = nanos(); 
      double optTime = (end - start) * 1e-9;
      double optmem =  memfactorgb / optTime;
      std::cout << "Transpose: " << optTime
              << "s, GB/S = " << optmem << "\n";
    }

    int main(){
       //First argument is the size of the matrix (Multiple of 8)
       TransposeBenchmark(8192);
    }

    //==========Benchmark output==========:
    67.1089 KB
    Transpose: 0.160259s, GB/S = 3.11995

## Code examples:
- How to use the makefile and extra compile flags:
    ```make
    //This cleans all .o files from the /build folder
    make clean
    //This builds the current project as is with ALL preprocessors disabled by default except AVX256
    make
    //You can explicitly disable/enable AVX256 optimizations (Currently supported by MatMul and MatrixTranspose)
    /*If left enabled or specified as enabled, the makefile will automatically
    add all required CXXFLAGS and LDFLAGS to the current commandline instance*/
    make USE_AVX256=0
    make USE_AVX256=1
    
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
   //Define a maximum amount of epochs and a squeezemax (not required, so can be set to 1)
   size_t epochmax = 100;
   size_t squeezemax = 1; 
   auto start = nanos(); 
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<float> dist(-50.0f, 100.0f);
   std::vector<float> inputVals;
   for(size_t i = 0; i < 10000; ++i) {
     inputVals.emplace_back(dist(gen));
   }
   std::vector<float> targetVals = {1.0f};
   float eta = 0.000001; 
   //Create a net object
   Neural::nn net;
   //The first layer input size has to be as big as the data set
   net.addLinear(inputVals.size(),1000);
   //Adds a relu layer
   net.addRelu(1000);
   net.addLinear(1000,100);
   //Adds a sigmoid layer
   net.addSigmoid(100);
   net.addLinear(100,1);
   //Attaches a loss function to the net as a make_unique to ensure ownership
   net.addLoss(std::make_unique<Neural::MSEloss>());
  
   std::vector<float> out; 
   float loss; 
              
   for(size_t epoch = 0; epoch < epochmax; ++epoch){
     for(size_t squeeze = 0; squeeze < squeezemax; ++squeeze){
       //Returns a vector from the Feed forward callback sequence
       out = net.Forward(inputVals);
       //Computes the loss based off of the target value vector
       loss = net.getLoss(targetVals);
       //Computes the loss gradient
       auto derivOut  = net.getGrad(targetVals);
       net.Backwards(derivOut);
       //Initiates the callback sequence to update the learning rate
       net.update(eta);
     }
     if(epoch % 10 == 0){
       //Simple output 
       printf("\033[47;30m | EPOCH = %i\033[m", (int)epoch);
       printf("\033[47;30m | LOSS = %f\033[m", loss);
       printf("\033[47;30m | OUTPUT[0] = %f\033[m", out[0]);
       printf("\033[47;30m | TARGET VAL = %i\033[m", (int)targetVals[0]);
       std::cout<<" [ ";
       //Loadbar is just a helper function to display a training progress bar
       loadbar(epoch);
     }
   }
   auto end = nanos(); 
   auto opttime = (end - start) * 1e-9;
   std::cout<<"\n\n";
   std::cout<< "||| Total training time: " << opttime << std::endl; 
   std::cout<< "||| Total EPOCHS: " << epochmax <<std::endl; 
   std::cout<< "||| Total SQUEEZE: " << squeezemax << std::endl;
   std::cout<< "||| Training data size: " << inputVals.size() << " data points" <<std::endl; 
   std::cout<<"\n"; 

  //Example output of this Neural network is as follows
  **EXECUTING**
  | EPOCH = 90 | LOSS = 50.924828 | OUTPUT[0] = 11.092059 | TARGET VAL = 1 [=====                ] 3.6%
 
  ||| Total training time: 1.11512
  ||| Total EPOCHS: 100
  ||| Total SQUEEZE: 1
  ||| Training data size: 10000 data points

- Tokenizing:
    ```c++
    //Initialize a tokenizer object 
    BPE::BPETokenizer tokenizer;
    std::string filePath = "src/TokenModels/DataSet.txt"; 
    std::ifstream infile {filePath};
    std::string trainingText {std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>()};
    //Define the number of merges for any possible detected grammar in the data set
    size_t numMerges = 5;
    std::cout << "Training BPE tokenizer with " << numMerges << " merges...\n";
    //Initialize the training sequence
    tokenizer.train(trainingText, numMerges);
    std::string testText = "I am testing out a large training data set for the tokenizer,
                            we will see if this works properly.";
    /*The training process will write two text files, bpe_vocab and bpe_merges. From the vocabulary set contained within them,
      We can now encode any input text with the same rules and grammar compression found in the initial data set.*/
    std::vector<BPE::g_tokenid> encodedIds = tokenizer.encode(testText);
    std::cout << "Encoded IDs for test text:\n";
    int idCount = 0; 
    for (const auto& id : encodedIds) {
      std::cout << "Encoded ID: " << id << " -> '" << tokenizer.decode({id}) << "'\n";
      idCount++;
      if(idCount == 10){
        std::cout<<"The rest of the encoded ID's output here ...\n" << std::endl;
        break;
      }
    }
    //Decode the encoded text ID's 
    std::string decodedText = tokenizer.decode(encodedIds);
    std::cout << "Decoded text: " << decodedText << std::endl;
    if (decodedText == testText) {
      std::cout << "***NOTE***: Encoding/decoding is lossless" << std::endl;
    } 
    else {
      std::cout << "***WARNING***: Encoding/decoding is not lossless" << std::endl;
    }
    //Saves current model vocabulary and merge rules so they can be re-used.
    tokenizer.saveModel("src/TokenModels/bpe_vocab.txt", "src/TokenModels/bpe_merges.txt");
    std::cout << "Model saved to files" << std::endl;
    tokenizer.printStats();

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

                            
## How It Works:
- ***A research paper and documentation will be written in due time.***
<br><br>
## Requirements
- C++17 or newer
- MinGW/GCC/G++
<br><br>
### Building 
- Compile and run using the provided Makefile:
    ```bash
    - Make sure to have 'Make' installed and 'GCC'/'G++' 
    - Set your C++ STL 'Includes' folder path in the CXXFLAGS section inside the Makefile
      (Only required if you do NOT have a config.yaml path file for your compiler)
    - Simply run 'make' directly in the command line from within the 'cppDL' folder.  
