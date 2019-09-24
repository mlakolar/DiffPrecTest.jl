using JLD
using DiffPrecTest
using DataFrames, CSV
using StatsPlots
using Distributions, Random, LinearAlgebra

function coverage(res::Vector{BootstrapEstimates}, α::Float64=0.95)
    NUM_REP = length(res)
    coverage = 0
    for rep=1:NUM_REP
        CI = simulCI(res[rep], α)
        coverage += all(0 .<= CI[:, 2]) * all(0 .>= CI[:, 1]) ? 1 : 0
    end
    coverage / NUM_REP
end

function coverage_studentized(res::Vector{BootstrapEstimates}, α::Float64=0.95)
    NUM_REP = length(res)
    coverage = 0
    for rep=1:NUM_REP
        CI = simulCIstudentized(res[rep], α)
        coverage += all(0 .<= CI[:, 2]) * all(0 .>= CI[:, 1]) ? 1 : 0
    end
    coverage / NUM_REP
end




function coverage_file_studentized(folderName, α::Float64=0.95;  NUM_REP::Int64=1000)
    coverage = 0
    for rep=1:NUM_REP
        res = load("$(folderName)_$(rep).jld", "boot_res")
        CI = simulCIstudentized(res, α)
        coverage += all(0 .<= CI[:, 2]) * all(0 .>= CI[:, 1]) ? 1 : 0
    end
    coverage / NUM_REP
end

function coverage_file(folderName, α::Float64=0.95;  NUM_REP::Int64=1000)
    coverage = 0
    for rep=1:NUM_REP
        res = load("$(folderName)_$(rep).jld", "boot_res")
        CI = simulCI(res, α)
        coverage += all(0 .<= CI[:, 2]) * all(0 .>= CI[:, 1]) ? 1 : 0
    end
    coverage / NUM_REP
end



coverageTable = zeros(3, 4)
sim = [8,9,10,11]
setting = [1,2,3]

for iSim = 1:4
    for iSetting = 1:3
        @show iSim, iSetting
        global sim
        global setting
        global coverageTable
        s1 = sim[iSim]
        s2 = setting[iSetting]
        folderName = "/scratch/midway2/mkolar/diffTest/sim$(s1)_o/res_$(s2)"
        coverageTable[iSetting, iSim] = coverage_file(folderName)
    end
end

@show coverageTable
