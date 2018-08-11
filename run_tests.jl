include("rich_method.jl")
f = open("test/emr/emr"*ARGS[1]*".txt")
Ws = Array{Any, 1}()
for line in readlines(f)
	W = map(x -> Array(x), rich_method(eval(parse(line))))
	push!(Ws, W)
end

f = open("test/emr.txt")
preWs = Array{Any, 1}()
for line in readlines(f)
	push!(preWs, eval(parse(line)))
end

pyWs = Array{Array{Array{Int64, 2}, 1}, 1}()
for preW in preWs
	W = map(x -> transpose(hcat(x...)), preW)
	push!(pyWs, W)
end

if length(Ws) != length(pyWs)
	println("length(Ws): $(length(Ws))")
	println("length(pyWs): $(length(pyWs))")
	println("typeof(Ws): $(typeof(Ws))")
	println("typeof(pyWs): $(typeof(pyWs))")
	error("length(Ws) != length(pyWs)")
end

cond = [true]
for (W, pyW) in zip(Ws, pyWs)
	if length(W) != length(pyW)
		print(W)
		print("\n\n")
		error("length(W) != length(pyW)")
	end
	for (w, pyw) in zip(W, pyW)
		if w != pyw
			println(w - pyw)
			println("\n\n")
			cond[1] = false
		end
	end
end

println(cond)

include("k_rich_method.jl")
f = open("test/kemr/kemr"*ARGS[1]*".txt")
Ws = Array{Any, 1}()
for line in readlines(f)
	toople = eval(parse(line))
	W = map(x -> Array(x), k_rich_method(toople[1], toople[2]))
	push!(Ws, W)
end

f = open("test/kemr.txt")
preWs = Array{Any, 1}()
for line in readlines(f)
	push!(preWs, eval(parse(line)))
end

pyWs = Array{Array{Array{Int64, 2}, 1}, 1}()
for preW in preWs
	W = map(x -> transpose(hcat(x...)), preW)
	push!(pyWs, W)
end

if length(Ws) != length(pyWs)
	error("length(Ws) != length(pyWs)")
end

cond = [true]
for (W, pyW) in zip(Ws, pyWs)
	if length(W) != length(pyW)
		print(W)
		print("\n\n")
		error("length(W) != length(pyW)")
	end
	for (w, pyw) in zip(W, pyW)
#		seiz = size(pyw)
#		vals = Array{Int64, 1}()
#		for j in 1:seiz[1]
##			for val in pyw[j, :]
#				push!(vals, val)
#			end
#		end
#		gnu = transpose(reshape(vals, seiz))
#		gnu = reshape(vals, (seiz[2], seiz[1]))
#		if w != gnu
		if w != pyw
			println("size(w): $(size(w))")
#			println("size(gnu): $(size(gnu))")
			println("size(pyw): $(size(pyw))")
			println("w:")
			for j in 1:seiz[2]
				println(w[j, :])
			end
			println("\ngnu:")
			for j in 1:seiz[2]
				println(gnu[j, :])
			end
			println("\n\n")
			cond[1] = false
		end
	end
end

println(cond)
