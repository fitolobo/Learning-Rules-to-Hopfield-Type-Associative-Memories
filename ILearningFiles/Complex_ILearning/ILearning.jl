module ILearning #Incremental Learning Module

# Packages
using LinearAlgebra
using Random
using Statistics
using Printf
using JLD2, FileIO
using ProgressMeter
rng = MersenneTwister(1234);

##########################################################################
### Activation Functions #################################################
##########################################################################

function continuous(x,a,params = nothing)
    # Compute the next state;
    if abs(a)>1e-10
        x = a/abs(a)
    end
    
    return x
    
end

function splitsign(x, a, Params = nothing)
    return sign(real(a)) + sign(imag(a))*im
end

function csign(x,a,K)
    phase_quanta = (round.(K*(2*pi.+angle.(a))./(2*pi))).%K
    z = exp.(2.0*pi*phase_quanta*im/K)
    return z
end


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
### Asynchronous Hopfield Neural Network
##########################################################################

function Asy(W, xinput,ActFunction,ActFunctionParams, it_max = 1.e3, verbose=false)

    Name = "Unit Asynchronous SplitSign"

    N = size(W,1)
    tau = 1.e-6

    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            # Compute the quaternion-valued activation potentials;
            a = dot(conj(W[ind[i],:]),x)

            # Compute the next state;
            x[ind[i]] = ActFunction(x[ind[i]],a,ActFunctionParams)

        end
        Error = norm(x-xold)
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

    
function noise(x,States,pr)
    
    xt = copy(x)
    
    ind = rand(rng,Float64,size(x)).< pr
    
    xt[ind] = rand(rng,States,sum(ind))
    
    return xt
    
end


function Multi_Estados()
        

    MultiStates_binary = [+1,-1]
    MultiStates_complex = [1.0 + 1.0*im, 1.0 - 1.0*im, -1.0 + 1.0*im, -1.0 - 1.0*im]           
            
    return MultiStates_binary,MultiStates_complex                 
end    
    
end #module
