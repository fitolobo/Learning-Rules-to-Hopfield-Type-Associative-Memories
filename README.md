# Learning-Rules-to-Hopfield-Type-Associative-Memories


Analysis, Comparisons and Construction of Learning Rules to Hopfield-Type Neural Networks in real and complex number systems. 

# Getting Started

This repository contain the Julia source-codes of Hopfield-Type learning rules on real and complex number systems. As described in the book chapter "Learning Rules to Hopfield Type Associative Memories" by Fidelis Zanetti and Rodolfo Anibal Lobo (see **Link**). The Jupyter-notebook of the computational experimens are also available in this repository.

In particular, we implemented the Correlation, Projection and Storkey learning rules, applying the Hopfield neural network as an associative memory storing synthetic patterns. For the complex case we explore two models, using *splitsign* and *csign* activation functions. 

# Usage

The main method for Storkey learning rules can be called in real case by
```julia
    W1 = RealStorkey.storkey_learning(U,ComplexStorkey.first)
    W2 = RealStorkey.storkey_learning(U,ComplexStorkey.second)
```
where the instruction ```first``` is the first order method and ```second``` the second order method. 
In the complex case by

 ```julia
    W1 = ComplexStorkey.storkey_learning(U,ComplexStorkey.first)
    W2 = ComplexStorkey.storkey_learning(U,ComplexStorkey.second)
 ```
 where the arguments ``` first ``` and ``` second ``` are analogous to the real functions.
 The projection and correlation rules, in both cases are called by
 
  ```julia
    Wc = RealStorkey.Correlation(U)
    Wp = RealStorkey.Projection(U)
 ```
 
 ```julia
    Wc = ComplexStorkey.Correlation(U)
    Wp = ComplexStorkey.Projection(U)
 ```
 The stored patterns have components in the binary set <img src="https://render.githubusercontent.com/render/math?math=%5C%7B%2B1%2C-1%5C%7D"> for real case. In the complex case you need to set the *resolution factor* <img src="https://render.githubusercontent.com/render/math?math=K"> in order to define the possible states for the neurons. In real case is obtained <img src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%20N%7D">, and in the complex case <img src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BC%7D%5E%7BN%5Ctimes%20N%7D"> both matrices symmetric and with zero diagonal terms. 


- **Fidelis Zanatti and Rodolfo Anibal Lobo** - *Federal Institute of Education, Science and Technology of Espírito Santo and University of Campinas, Brazil 2020.*
