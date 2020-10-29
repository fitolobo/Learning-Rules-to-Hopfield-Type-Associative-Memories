module ComplexStorkey

# pacotes utilizados
using LinearAlgebra
using Random
using Statistics
using Printf
using Quaternions
using JLD2, FileIO
using ProgressMeter
rng = MersenneTwister(1234);

##########################################################################
### Activation Functions #################################################
##########################################################################

function continuous(a,params = nothing)
    
    uni = a/abs(a)
    
    return uni
end

function splitsign(a, Params = nothing)

    z = real(a)/abs(real(a)) + (imag(a)/abs(imag(a)))*im

    return z

end

function csign(x,K)
    phase_quanta = (round.(K*(2*pi.+angle.(x))./(2*pi))).%K
    z = exp.(2.0*pi*phase_quanta*im/K)
    return z
end
##########################################################################

##########################################################################

##########################################################################


##########################################################################
### General Storing Rules ################################################
##########################################################################

function Correlation(U)
    
   N = size(U)[1] 
    
   Uc = (1/N)*U'
    
   W = U*Uc
    
   W = W - Diagonal(Diagonal(W));
    
    
   return W 
    
end


function Projection(U)
   
   N = size(U,1)  
    
   Up = (inv((U'*U))*U')/N
    
   W = U*Up
    
   W = W - Diagonal(Diagonal(W));
    
    
   return W 
    
end

##########################################################################

##########################################################################

##########################################################################

##########################################################################
### Storkey Storing Rules ################################################
##########################################################################

##########################################################################
### Main #################################################################
##########################################################################

function storkey_learning(U,order)
    # The memories are given by the columns
    row,col = size(U)
    
    # First W matrix
    W_new = zeros(Complex,row,row)
    W_new = (1/row) * U[:,1]*U[:,1]'
    
    for mu=2:col
        
        W_mu = zeros(Complex,row,row)
        
        W_old = copy(W_new)
        
        W_new +=  order(U,W_old,W_mu,mu,row)
        
    end
    
    return W_new

end

##########################################################################
### First Order ##########################################################
##########################################################################

function first(U,W_old,W_mu,mu,row)
    
    # Chamo uma vez só o elemento
    U_mu = U[:,mu]
    
    
  @views @inbounds   for j=1:row
        U_j_mu = U[j,mu]
        
    @views @inbounds     for i=j:row 
            
            U_i_mu = U[i,mu]
            s = 0.0+0.0*im
            s += U_i_mu *conj(U_j_mu)
            s -= conj(local_field_opt(W_old,U_mu,j,i,row))*U_i_mu
            s -= local_field_opt(W_old,U_mu,i,j,row)*conj(U_j_mu)
            s *= 1/row    

            W_mu[i,j] += s
            W_mu[j,i]  = conj(W_mu[i,j])

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
    
    
 @views @inbounds    for j=1:row
        U_j_mu = U[j,mu]
        
   @views @inbounds      for i=j:row 
            
            U_i_mu = U[i,mu]
            
            s = 0.0+0.0*im

            # Putting this value in the new matrix
            v1  = local_field_opt(W_old,U_mu,i,j,row)
            v2  = local_field_opt(W_old,U_mu,j,i,row)
            s += U_i_mu*conj(U_j_mu) 
            s -= U_i_mu*conj(v2)
            s -= v1*conj(U_j_mu)
            s += v1*conj(v2)
            s *= 1/row    

            W_mu[i,j]  += s
            W_mu[j,i]  = conj(W_mu[i,j])

        end  
    end

    
    return W_mu
end

##########################################################################
### Local Field ##########################################################
##########################################################################

function local_field_opt(W_old,U,i,j,row)
    
    hij = 0.0+0.0*im
   
    @inbounds @simd for k=1:row
        
        hij += ifelse((k!=i)&(k!=j), W_old[i,k]*U[k], 0.0+0.0*im)
        
    end
    
   return  hij
    
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
            
            if abs(real(a)*imag(a))>1e-10
                x[ind[i]] = ActFunction(a,ActFunctionParams)
            end

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

    
    
    
function AsyCV(W, xinput,ActFunction,ActFunctionParams, it_max = 1.e3, verbose=false)

    Name = "Unit Asynchronous Complex Sign"

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
            if abs(a)>1e-10
                x[ind[i]] = ActFunction(a,ActFunctionParams)
            end

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
        
        
function AsyComplexSign(W, xinput,ActFunction,ActFunctionParams, it_max = 1.e3, verbose=false)

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
            x[ind[i]] = ActFunction(a,ActFunctionParams)


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
    
function noise_prob(x,MultiStates,pr)
    
    verdaderos = 1.0 * (rand(rng,Float64,size(x)) .< pr)   
    
    mantem = -1.0*((verdaderos) .- 1.0)   
    
    novo = rand(rng,MultiStates,size(x))
    
    xout = -1.0.*verdaderos.*novo + mantem .*x
    
    return xout
    
end


function Multi_Estados()
            
    qstates = 0:1:15;
    qstates_array = zeros(16,4)
    MultiStates_quat = Array{Quaternion{Float64}}(undef,16,1)
    for i=1:16
        qstates_array[i,:] = 2 .* digits(UInt(qstates[i]), base=2, pad=4) .- 1
        MultiStates_quat[i,1] = Quaternion(qstates_array[i,1],qstates_array[i,2],qstates_array[i,3],qstates_array[i,4])
    end

    MultiStates_binary = [+1,-1]
    MultiStates_complex = [1.0 + 1.0*im, 1.0 - 1.0*im, -1.0 + 1.0*im, -1.0 - 1.0*im]           
            
    return MultiStates_binary,MultiStates_complex,MultiStates_quat                 
end    
    
end #module
