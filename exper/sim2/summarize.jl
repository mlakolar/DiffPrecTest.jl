using JLD
using DiffPrecTest
using DataFrames, DataFramesMeta, CSV
using LaTeXStrings, LaTeXTabulars


cd("exper/sim2")

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25)]
methodArr = ["Sym-N", "Asym-N", "YinXia", "Sym-B", "Asym-B", "O-Sym-N", "O-Asym-N", "O-Sym-B", "O-Asym-B"]

df = DataFrame(p = Int[], row = Int[], col = Int[], method=String[], bias=Float64[], coverage=Float64[], lenCI=Float64[])

for ip=1:2
  for iElem=1:3
    res = load("results/sim2_res_$(ip)_$(iElem).jld", "results")

    for j in 1:9
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

### create a table 

latex_tabular("sim2_table.tex",
  Tabular("lllccc"),
  [
    Rule(:top),
    ["", "", "", "Coverage", "Length", L"{\rm Bias} \times 10^3"],
    Rule(:mid),
    vcat([ 
      hcat(
        (a = fill("", 9, 1); 
        b = latexstring("p = $(pArr[ip])");
        a[1] = "\\multirow{9}{*}{$(b)}"; 
        a
        ),  
        vcat([
          hcat(
            (a = fill("", 3, 1); 
            b = latexstring("\\Delta_{$(elemArr[iElem][1]),$(elemArr[iElem][2])} = 0");
            a[1] = "\\multirow{3}{*}{$(b)}"; 
            a
            ),
            Matrix{Any}(
              @linq df |>
              where( (:method .== "Sym-N") .| (:method .== "YinXia") .| (:method .== "O-Sym-N"), 
                    :p .== pArr[ip], 
                    :row .== elemArr[iElem][1], :col .== elemArr[iElem][2]) |>  select(:method, 
                    coverage = convert.(Int, :coverage * 1000), 
                    lenCI = round.(:lenCI, digits=3), 
                    bias = round.(:bias * 1000, digits=1))   
            )
          ) 
          for iElem=1:3
          ]...
        )
      )
      for ip=1:2
    ]...),
    Rule(:bottom)
  ]
)