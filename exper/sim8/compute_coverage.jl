using DiffPrecTest
using JLD

coverage = 0
coverage_stud = 0

fName = ARGS[1]
alpha = parse(Float64, ARGS[2])

NUM_REP = 1000

res = load(fName, "results")

for rep=1:NUM_REP       
    global coverage 
    global coverage_stud

    CI = simulCI(res[rep], alpha)
    coverage += all(0 .<= CI[:, 2]) * all(0 .>= CI[:, 1]) ? 1 : 0

    CI = simulCIstudentized(res[rep], alpha)
    coverage_stud += all(0 .<= CI[:, 2]) * all(0 .>= CI[:, 1]) ? 1 : 0
end

@show coverage / NUM_REP
@show coverage_stud / NUM_REP