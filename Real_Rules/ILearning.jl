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
function Projection(U)
   
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


function train(order,U, Win = nothing)
    
    # The memories are given by the columns
    row,col = size(U)
    
    #if Win == nothing
       # First W matrix
    W_new = narg(Win,row)
    #   return W_new   
    #else
    #   W_new = copy(Win)
    #    return W_new
    #end
    
       @views @inbounds for mu=1:col
        
        W_mu = zeros(row,row)
        
        W_old = copy(W_new)
        
        W_new +=  order(U,W_old,W_mu,mu,row)
        
    end
    
    return W_new-Diagonal(Diagonal(W_new))    

end


##########################################################################
### First Order ##########################################################
##########################################################################

function first(U,W_old,W_mu,mu,row)
    
    # Chamo uma vez só o elemento
    U_mu = U[:,mu]
    
    
    @views @inbounds for j=1:row
        U_j_mu = U[j,mu]
        
        for i=j:row 
            
            U_i_mu = U[i,mu]
            
            # Putting this value in the new matrix
            
            s = 0.0
            # Putting this value in the new matrix
            
             s += U_i_mu *U_j_mu
             s -= U_i_mu *local_field_opt(W_old,U_mu,j,i,row)
             s -= local_field_opt(W_old,U_mu,i,j,row)*U_j_mu
             s *= 1/row    

            W_mu[i,j] += s
            W_mu[j,i]  = W_mu[i,j]
            #W_mu[i,i]  = 0.0
        end  
    end

    
    return W_mu
end

##########################################################################
### Second Order##########################################################
##########################################################################

function second(U,W_old,W_mu,mu,row)
    
    # Chamo uma vez só o elemento
    U_mu = U[:,mu]
    
    
  @views @inbounds  for j=1:row
        U_j_mu = U[j,mu]
        
        for i=j:row 
            
            U_i_mu = U[i,mu]
            
            s = 0.0

            # Putting this value in the new matrix
            v1  = local_field_opt(W_old,U_mu,i,j,row)
            v2  = local_field_opt(W_old,U_mu,j,i,row)
            s += U_i_mu*U_j_mu 
            s -= U_i_mu*v2
            s -= v1*U_j_mu
            s += v1*v2
            s *= 1/row    

            W_mu[i,j]  += s
            W_mu[j,i]  = W_mu[i,j]
            #W_mu[i,i]  = 0.0 
        end  
    end

    
    return W_mu
end

##########################################################################
### Local Field ##########################################################
##########################################################################

function local_field_opt(W_old,U,i,j,row)
    
    hij = 0.0
    
    @inbounds @simd for k=1:row
        
        hij += ifelse((k!=i)&(k!=j), W_old[i,k]*U[k], 0.0)
        
    end
    
   return  hij
    
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

    Error= N -1
    
    while (Error<N) && (it<it_max)
 
        it = it+1
        
        for i = 1:N       
            # Compute the next state
            a = dot(W[i,:],x)
            
            if abs(a)> 0 
                x[i] = a /abs(a)  
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


end 