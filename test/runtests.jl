tests = [
  "util",
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
    t = @elapsed include(f)
    println("done (took $t seconds).")
end
