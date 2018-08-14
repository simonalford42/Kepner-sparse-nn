# Examples A and B give same DNN topology for both formats.
#= Explicit format (more precise, but not default):
Example A: 
Ns = [[7,7,7], [7]]
B = [3, 2, 6, 9, 1]
D = [1, 1, 1, 1]]

Example B:
Ns = [[3,3]]
B = [1, 2, 1]
D = [1, 1]

Example C:
Ns = [[13,7,10]]
B = [3, 2, 6, 9]
D = [1, 2, 3]

Example D:
Ns = [[8,8,8,8], [4,8,16,8], [16,8,8,16], [8,16]]
B = [1, 4, 2, 3, 1, 1, 1, 2, 3, 4, 2, 3, 3, 2, 1]
D = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
=#

### BEGIN MULTI-LINE COMMENT
#= Remove this multi-line comment if explicit format preferred.
Ns = [[10, 10, 10], [10]]
B = [1, 2, 3, 2, 1]
D = [1, 1, 1, 1]
=#
### END MULTI-LINE COMMENT

#= Reduced format (default):
Example A:
N = [1029, 686, 2058, 3087, 343]
D = [14, 42, 63, 7]

Example B:
N = [9, 18, 9]
D = [6, 3]

Example C:
N = [900, 1800, 900, 1800, 2700, 4500, 2700, 1800, 900]
D = [10, 20, 9, 20, 30, 45, 30, 20, 9]

Example D:
N = [1820, 3640, 910, 5460, 3640]
D = [14, 8, 5, 78, 28]
=# 

### IF DOING EXPLICIT FORMAT, BEGIN MULTI-LINE COMMENT HERE
N = [1000, 2000, 3000, 2000, 1000]
D = [20, 30, 20, 10]

# Array to store indices that begin new radix systems
inds = Array{Int32, 1}()

# Check that N, D define a valid topology,
# and convert to explicit format
gN = gcd(N)
# Make B (explicit format)
B = map(x -> div(x, gN), N)
DD = map(x -> div(x[1], x[2]), zip(D, B[2:end]))
fac = 1
start_ind = 1
push!(inds, start_ind)
for (j, d) in enumerate(DD)
	fac *= d
	if fac > gN
		factorization = map(x -> "("*string(x)*")", DD[start_ind:j])
		error("DD-indices $(start_ind) to $j; $(gN) != "*prod(factorization)*".")
	elseif fac == gN
		start_ind = j+1
		push!(inds, start_ind)
		fac = 1
	end
end
push!(inds, length(D)+1)
if gN % fac != 0
	factorization = map(x -> "("*string(x)*")", DD[start_ind:end])
	error("DD-indices $(start_ind) to end; $(gN) % "*prod(factorization)*" != 0.")
end

# Make Ns (explicit format)
if length(inds) == 1
	Ns = [DD]
else
	Ns = map(j -> DD[inds[j]:inds[j+1]-1], 1:(length(inds)-1))
end

# Make D (explicit format)
D = ones(Int32, length(DD))

### IF DOING EXPLICIT FORMAT, END MULTI-LINE COMMENT HERE


NN = gN
M = sum(map(x -> length(x), Ns))

for (j, d) in enumerate(D)
	if (gcd(B[j], B[j+1]) % d) != 0
		error("D[$j] must divide gcd(B[$j], B[$j+1]).")
	end
end

# Construct weight matrices for the case B = ones(Int32, M+1),
# D = ones(Int32, M)
I = eye(Int32, NN)
preW = Array{Array{Int32, 2}, 1}() # list of w_i's 
for N in Ns
	pv = 1
	for d in N
        # preW starts out empty. This part does the sum_j^n-1 (P^jvi)
        # pv is the 1,8,64 thing. j goes from 0 to 9. 
		push!(preW, sum([circshift(I, (j*pv, 0)) for j in 0:(d-1)]))
        	pv *= d
	end
end

# Construct matrices A for Kronecker products A x B
krons = Array{Array{Int32, 2}, 1}()
for j in 1:M
	kMatD = zeros(Int32, (B[j], B[j+1]))
	g = gcd(B[j], B[j+1])
	g_to_d = div(g, D[j])
	preb = div(B[j], g)
	postb = div(B[j+1], g)
	eyeD = eye(Int32, g)
	kMatD = sum([circshift(eyeD, (0, dd*g_to_d)) for dd in 0:D[j]-1])
	kMat = kron(ones(Int32, (preb, postb)), kMatD)
	push!(krons, kMat)
end

# Construct actual weight matrices (the final layers)
W = [kron(kMat, prew) for (kMat, prew) in zip(krons, preW)]

# Mutliply adjacency matrices
Wprod = prod(W)

# Run test
test1 = maximum(Wprod) == minimum(Wprod)
numpaths = maximum(Wprod)

# Print values
println("Test: $(test1)")
println("Number of Paths: $(numpaths)")
