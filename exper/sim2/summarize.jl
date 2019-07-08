using JLD
using DiffPrecTest
using DataFrames, CSV

pArr = [100, 200, 500]
elemArr = [(5,5), (8, 7), (50, 25)]
methodArr = ["Our-N", "YinXia", "Our-B", "Oracle-N", "Oracle-B"]

df = DataFrame(p = Int[], row = Int[], col = Int[], method=String[], bias=Float64[], coverage=Float64[], lenCI=Float64[])

for ip=1:3
  for iElem=1:3
    res = load("../sim2_res_$(ip)_$(iElem).jld", "results")
    
    for j in 1:5
      global pArr
      global elemArr
      global methodArr
      global df

      sr = computeSimulationResult([res[j, i] for i=1:1000], 0.)
      push!(df, [pArr[ip], elemArr[iElem][1], elemArr[iElem][2], methodArr[j], sr.bias, sr.coverage, sr.lenCoverage])
    end
  end
end

CSV.write("sim2.csv", df)
