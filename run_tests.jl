include("rich_method.jl")
f = open("emr_vars.txt")
Ws = Array{Any, 1}()
for line in readlines(f)
	push!(Ws, rich_method(eval(parse(line))))
end

f = open("test/emr.txt")
preWs = Array{Any, 1}()
for line in readlines(f)
	push!(preWs, eval(parse(line)))
end

pyWs = Array{Array{Array{Int64, 2}, 1}, 1}()
for preW in preWs
	m = isqrt(length(preW[1]))
	W = map(x -> hcat(x...), preW)
	push!(pyWs, W)
end

println(Ws == pyWs)
