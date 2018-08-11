using Permutations

function conv(x::Int, j::Int, pv::Int, NN::Int)
	pre = mod((x-1-j*pv), NN) + 1
	return pre
end

function extended_mixed_radix_network(Ns::Array{Array{Int, 1}, 1})
	NN = prod(Ns[1])

	W = Array{SparseMatrixCSC{Int32, Int32}, 1}()

	for N in Ns
		pv = 1
		for d in N
			preW = Array{SparseMatrixCSC{Int32, Int32}, 1}()
			for j in 0:(d-1)
				pp = conv.(1:NN, j, pv, NN)
				p = Permutation(pp)
				push!(preW, sparse(p))
			end
			wubs = zero(preW[1])
			for wub in preW
				wubs += wub
			end
			push!(W, wubs)
        		pv *= d
		end
	end

	return W
end

function kronecker_emr_network(Ns::Array{Array{Int, 1}, 1}, B::Array{Int, 1}, D::Union{Array{Int, 1}, Void}=nothing)
	M = sum(length.(Ns))
	gs = map(x -> gcd(B[x], B[x+1]), 1:M)
	if D == nothing
		D = copy(gs)
	end

	for (j, d) in enumerate(D)
		if gcd(B[j], B[j+1]) % d != 0
			error("D[$j] must divide gcd(B[$j], B[$j+1]).")
		end
	end

	preW = extended_mixed_radix_network(Ns)

        W = Array{SparseMatrixCSC{Int32, Int32}, 1}()
        for j in 1:length(preW)
		gd = div(gs[j], D[j])
		Gamma = zeros(Int32, (gs[j], gs[j]))
		for k in 0:(D[j]-1)
			pp = conv.(1:gs[j], k, gd, gs[j])
			p = Permutation(pp)
			Gamma += Array(p)
		end
		BG = kron(ones(Int64, (B[j], B[j+1])), Gamma)
                push!(W, kron(BG, preW[j]))
        end

        return W
end
