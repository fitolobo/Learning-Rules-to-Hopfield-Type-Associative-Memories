# Learning-Rules-to-Hopfield-Type-Associative-Memories


Analysis, Comparisons and Construction of Learning Rules to Hopfield-Type Neural Networks in real and complex number systems. 

# Getting Started

This repository contain the Julia source-codes of Hopfield-Type learning rules on real and complex number systems. As described in the work "Learning Rules to Hopfield-Type Associative Memories in Real and Complex Domains" by Fidelis Zanetti and Rodolfo Anibal Lobo. The Jupyter-notebook of the computational experimens are also available in this repository.

In particular, we implemented the Correlation, Projection and Storkey learning rules, applying the Hopfield neural network as an associative memory storing synthetic patterns. For the complex case we explore two models, using *splitsign* and *csign* activation functions. 

# Usage

The main method for Storkey learning rules can be called in real case by
```julia
    W1 = ILearning.train(ILearning.first,U,nothing)
    W2 = ILearning.train(ILearning.second,U,nothing)
```
where the instruction ```first``` is the first order method and ```second``` the second order method. The third argument let us intialize with a non null matrix previously stored, for example:
```julia
    ### Create a boolean random matrix
    N = 200;
    P1 = 10
    U1 = 2*rand(rng,Bool,(N,P1)).-1;
    
    ### Storing U1 using first order storkey rule (initializing Win = 0)
    W = ILearning.train(ILearning.first,U1,nothing);
    
    ### Create other boolean random matrix
    P2 = 10
    U2 = 2*rand(rng,Bool,(N,P2)).-1;
    
    ### Storing U2 using first order storkey rule (initializing Win = W) 
    W1 = ILearning.train(ILearning.first,U2,W);
```
Analogously in the complex case, we have the storkey learning rules

 ```julia
    W1 = ILearning.train(ILearning.first,U,nothing)
    W2 = ILearning.train(ILearning.second,U,nothing)
 ```
 where the arguments ``` first ``` and ``` second ``` are analogous to the real functions, and the third argument let us intialize with a non null matrix previously stored using the same storage rule. The projection and correlation rules, in both cases are called by
 
  ```julia
    Wc = ILearning.Correlation(U,nothing);
    Wp = ILearning.Projection(U)
 ```
 In this case, the correlation rule also permits initialize with a non null matrix previously stored using the same storage rule. 

 The stored patterns have components in the binary set <img src="https://render.githubusercontent.com/render/math?math=%5C%7B%2B1%2C-1%5C%7D"> for real case. In the complex case you need to set the *resolution factor* <img src="https://render.githubusercontent.com/render/math?math=K"> in order to define the possible states for the neurons. In real case is obtained <img src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%20N%7D">, and in the complex case <img src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BC%7D%5E%7BN%5Ctimes%20N%7D"> both matrices symmetric and with zero diagonal terms. 


- **Fidelis Zanatti, Rodolfo Anibal Lobo and Marcos Eduardo Valle** - *Federal Institute of Education, Science and Technology of Espírito Santo and University of Campinas, Brazil 2020.*
