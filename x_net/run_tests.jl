include("x_net.jl")
f = open("test/emr/emr"*ARGS[1]*".txt")
Ws = Array{Any, 1}()
for line in readlines(f)
	W = map(x -> Array(x), emr_net(eval(parse(line))))
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

f = open("test/kemr/kemr"*ARGS[1]*".txt")
Ws = Array{Any, 1}()
for line in readlines(f)
	toople = eval(parse(line))
	W = map(x -> Array(x), kemr_net(toople[1], toople[2]))
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

if 1 or length(Ws) != length(pyWs)
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
			println("size(w): $(size(w))")
			println("size(pyw): $(size(pyw))")
			cond[1] = false
		end
	end
end

println(cond)
