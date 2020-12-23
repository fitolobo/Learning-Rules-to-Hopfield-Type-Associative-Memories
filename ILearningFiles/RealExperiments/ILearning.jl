module ILearning #incremental learning module

# pacotes utilizados
using LinearAlgebra
using Random
using Statistics
using Printf
using ProgressMeter
rng = MersenneTwister(1234);


##########################################################################
### General Storing Rules ################################################
##########################################################################

function Correlation(U,Win = nothing)
    
   row, col = size(U)
        
   W = narg(Win,row)
    
   for i=1:col
    
       W += (1/row)*U[:,i]*U[:,i]'
        
   end
    
   W = W - Diagonal(Diagonal(W));
    
    
   return W 
    
end

### Escribir de forma incremental
function Projection(U, Win = nothing, Ain = nothing)
    
   row, col = size(U)
        
   if Win == nothing
        W = zeros(row,row);
        A = I(row)
   else
        W = copy(Win)
        A = copy(Ain)
   end
        
   for i=1:col
       z = A*U[:,i];
       zz = real(z'*z)
       if zz>1.e-7
           A = A - (z*z')/zz;
           v = W*U[:,i]
           W += (U[:,i]-v)*(z'/zz);
       end
   end
   return W, A
end

function Projection_NonIncremental(U)
   
   N = size(U,1)  
    
   Up = (inv((U'*U))*U')/N
    
   W = U*Up
    
   W = W - Diagonal(Diagonal(W));
    
    
   return W 
    
end

##########################################################################
### Storkey Storing Rules ################################################
##########################################################################

##########################################################################
### Main #################################################################
##########################################################################
function narg(Win,row)
  
    if Win == nothing
       # First W matrix
        W_new = zeros(row,row)
        return W_new   
    else
        W_new = copy(Win)
        return W_new
    end
    
end


##########################################################################
### Asynchronous Hopfield Neural Network
##########################################################################

function Asy(W, xinput, it_max = 1.e3, verbose=false)

    Name = "Asynchronous Hopfield Neural Network"

    N = size(W,1)

    # Initialization
    x = copy(xinput)
    xold = copy(x)
    
    it = 0    

    Error= N-1
    
    while (Error<N) && (it<it_max)
 
        it = it+1
        
        for i = 1:N       
            # Compute the next state
            a = dot(W[i,:],x)
            if abs(a)> 0 
                x[i] = sign(a)  
            end
            
        end
        
        Error = dot(x,xold)
        xold = copy(x)
        
    end

    if verbose == true

        if it_max<=it
            println(Name," failed to converge in ",it_max," iterations.")
        else
            println(Name," converged in ",it," iterations using asynchronous update.")    
        end
    end

    return x
end


function noise(x,pr)
    
    xt = copy(x)
    
    ind = rand(rng,Float64,size(xt)).< pr
    
    xt[ind] = -xt[ind]
    
    return xt
    
end




end 
