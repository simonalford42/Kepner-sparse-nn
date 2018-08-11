using Permutations

function k_rich_method(Ns::Array{Array{Int, 1}, 1}, B::Array{Int, 1})
	b = sum(map(x -> length(x), Ns))
	D = ones(Int32, b)

	gN = prod(Ns[1])

	NN = gN
	M = sum(map(x -> length(x), Ns))

	for (j, d) in enumerate(D)
		if (gcd(B[j], B[j+1]) % d) != 0
			error("D[$j] must divide gcd(B[$j], B[$j+1]).")
		end
	end

	# Construct weight matrices for the case B = ones(Int32, M+1),
	# D = ones(Int32, M)
	preW = Array{SparseMatrixCSC{Int32, Int32}, 1}()
	function conv(x::Int, j::Int, pv::Int)
		pre = mod((x-1-j*pv), NN) + 1
		return pre
	end

	for N in Ns
		pv = 1
		for d in N
			prepreW = Array{SparseMatrixCSC{Int32, Int32}, 1}()
			for j in 0:(d-1)
				pp = conv.(1:NN, j, pv)
				p = Permutation(pp)
				push!(prepreW, sparse(p))
			end
			wubs = zero(prepreW[1])
			for wub in prepreW
				wubs += wub
			end
			push!(preW, wubs)
        		pv *= d
		end
	end

	W = Array{SparseMatrixCSC{Int32, Int32}, 1}()
	for j in 1:length(preW)
		push!(W, kron(ones(Int64, (B[j], B[j+1])), preW[j]))
	end

	return W
end
