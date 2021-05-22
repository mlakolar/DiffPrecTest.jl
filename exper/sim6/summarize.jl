using JLD
using DiffPrecTest
using DataFrames, DataFramesMeta, CSV
using Random, LinearAlgebra, Distributions
using LaTeXStrings, LaTeXTabulars

cd("exper/sim6")

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]
methodArr = ["Sym-N", "Asym-N", "YinXia", "Sym-B", "Asym-B", "O-Sym-N", "O-Asym-N", "O-Sym-B", "O-Asym-B"]

df = DataFrame(p = Int[], row = Int[], col = Int[], trueValue=Float64[], method=String[], bias=Float64[], coverage=Float64[], lenCI=Float64[])

trueValue = zeros(Float64, 2, 5)

for ip=1:2
  for iElem=1:5
    p = pArr[ip]
    Random.seed!(7658)

    # generate model
    Ωx = Matrix{Float64}(I, p, p)
    for l=1:p-1
      Ωx[l  , l+1] = 0.3
      Ωx[l+1, l  ] = 0.3
    end
    for l=1:p-2
      Ωx[l  , l+2] = 0.2
      Ωx[l+2, l  ] = 0.2
    end
    Ωy = Matrix{Float64}(I, p, p)
    for l=1:p-1
      Ωy[l  , l+1] = 0.3
      Ωy[l+1, l  ] = 0.3
    end
    for l=1:p-2
      Ωy[l  , l+2] = -0.1
      Ωy[l+2, l  ] = -0.1
    end
    Δ = Ωx - Ωy
    # generate Delta
    d = Vector{Float64}(undef, p)
    rand!(Uniform(0.5, 2.5), d)
    d .= sqrt.(d)
    D = Diagonal(d)
    Σx = inv(Symmetric(D * Ωx * D))
    Σy = inv(Symmetric(D * Ωy * D))

    tΔ = D * Δ * D

    res = load("results/sim6_res_$(ip)_$(iElem).jld", "results")

    ri, ci = elemArr[iElem]
    indE = (ci - 1) * p + ri

    trueValue[ip, iElem] = tΔ[ri, ci]    

    for j in 1:9
      sr = computeSimulationResult([res[j, i] for i=1:1000], tΔ[indE])
      push!(df, [pArr[ip], elemArr[iElem][1], elemArr[iElem][2], tΔ[indE], methodArr[j], sr.bias, sr.coverage, sr.lenCoverage])
    end
  end
end

CSV.write("sim6.csv", df)



### create a table 

latex_tabular("sim6_table.tex",
  Tabular("lllccc"),
  [
    Rule(:top),
    ["", "", "", "Coverage", "Length", L"{\rm Bias} \times 10^3"],
    Rule(:mid),
    vcat([ 
      hcat(
        (a = fill("", 15, 1); 
        b = latexstring("p = $(pArr[ip])");
        a[1] = "\\multirow{15}{*}{$(b)}"; 
        a
        ),  
        vcat([
          hcat(
            (a = fill("", 3, 1); 
            b = latexstring("\\Delta_{$(elemArr[iElem][1]),$(elemArr[iElem][2])} = $(trueValue[ip, iElem])");
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
          for iElem=1:5
          ]...
        )
      )
      for ip=1:2
    ]...),
    Rule(:bottom)
  ]
)
