# Learning-Rules-to-Hopfield-Type-Associative-Memories


Analysis, Comparisons and Construction of Learning Rules to Hopfield-Type Neural Networks in real and complex number systems. 

# Getting Started

This repository contain the Julia source-codes of Hopfield-Type learning rules on real and complex number systems. As described in the work "Learning Rules to Hopfield-Type Associative Memories in Real and Complex Domains" by Fidelis Zanetti, Rodolfo Anibal Lobo and Marcos Eduardo Valle. The Jupyter-notebook of the computational experimens are also available in this repository.

In particular, we implemented the Correlation, Projection and Storkey learning rules, applying the Hopfield neural network as an associative memory storing synthetic patterns. For the complex case we explore two models, using *splitsign* and *csign* activation functions. 

# Usage
The main module of incremental learning in real and complex case is called ```ILearning.jl```. The main method for Storkey learning rules can be called in real case by
```julia
    W1 = Storkey1(U,nothing)
    W2 = Storkey2(U,nothing)
```
where the function ```Storkey1``` is the first order method and ```Storkey2``` the second order method. The third argument let us intialize with a non null matrix previously stored, for example:
```julia
    ### Create a boolean random matrix
    N = 200;
    P1 = 10
    U1 = 2*rand(rng,Bool,(N,P1)).-1;
    
    ### Storing U1 using first order storkey rule (initializing Win = 0)
    W = Storkey1(U1,nothing);
    
    ### Create other boolean random matrix
    P2 = 10
    U2 = 2*rand(rng,Bool,(N,P2)).-1;
    
    ### Storing U2 using first order storkey rule (initializing Win = W) 
    W1 = Storkey2(U2,W);
```
Analogously in the complex case, we have the storkey learning rules

 ```julia
    W1 = Storkey1(U,nothing)
    W2 = Storkey2(U,nothing)
 ```
 where the functions ``` Storkey1 ``` and ``` Storkey2``` are the complex version of these learning rules, and the third argument let us intialize with a non null matrix previously stored using the same storage rule. The projection and correlation rules, in both cases are called by
 
  ```julia
    Wc = ILearning.Correlation(U,nothing);
    Wp = ILearning.Projection(U, Win, Ain)
 ```
 In this case, the correlation and projection rules could also be initialized with a non null matrix.

 The stored patterns have components in the binary set <img src="https://render.githubusercontent.com/render/math?math=%5C%7B%2B1%2C-1%5C%7D"> for real case. In the multistate complex case you need to set the *resolution factor* <img src="https://render.githubusercontent.com/render/math?math=K"> in order to define the possible states for the neurons, and when is used the splitsign activation function the componentes belongs to the set <img src="https://render.githubusercontent.com/render/math?math=%5C%7B%5Cpm%201%5Cpm%20%5Cmathbf%7Bi%7D%5C%7D">. In real case is obtained <img src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%20N%7D">, and in the complex case <img src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BC%7D%5E%7BN%5Ctimes%20N%7D"> both matrices symmetric and with zero diagonal terms. 


- **Fidelis Zanatti, Rodolfo Anibal Lobo and Marcos Eduardo Valle** - *Federal Institute of Education, Science and Technology of Espírito Santo and University of Campinas, Brazil 2020.*
