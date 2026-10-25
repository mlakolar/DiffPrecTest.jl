# Helpers for the summaries of the coverage simulations (sims 1-4, 6 and 7).
using DiffPrecTest
using DataFrames
using JLD2
using LaTeXStrings, LaTeXTabulars

# methods in the order of est in the simulation scripts
methodArr = ["Sym-N", "Asym-N", "YinXia", "Sym-B", "Asym-B", "O-Sym-N", "O-Asym-N", "O-Sym-B", "O-Asym-B"]

newSummary() = DataFrame(p = Int[], row = Int[], col = Int[], trueValue=Float64[], method=String[], bias=Float64[], coverage=Float64[], lenCI=Float64[])

# add the bias, coverage and length of the confidence intervals of every method
# for one setting, from a file written by collect_results.jl
function summarizeSetting!(df, file, p, (row, col), trueValue)
  res = load(file, "results")
  for j in eachindex(methodArr)
    sr = compute_simulation_result([res[j, i] for i=1:size(res, 2)], trueValue)
    push!(df, (p, row, col, trueValue, methodArr[j], sr.bias, sr.coverage, sr.len_coverage))
  end
  df
end

# LaTeX table with coverage (× 1000), length and bias (× 1000) of the methods
# for every dimension in pArr and entry in elemArr; trueValue[ip, iElem] is the
# value of the entry
function coverageTable(filename, df, pArr, elemArr, trueValue; methods=["Sym-N", "YinXia", "O-Sym-N"])
  m = length(methods)
  lines = Any[Rule(:top), ["", "", "", "Coverage", "Length", L"{\rm Bias} \times 10^3"], Rule(:mid)]
  for ip in eachindex(pArr), iElem in eachindex(elemArr)
    row, col = elemArr[iElem]
    v = trueValue[ip, iElem]
    for (k, method) in enumerate(methods)
      r = only(eachrow(df[(df.p .== pArr[ip]) .& (df.row .== row) .& (df.col .== col) .& (df.method .== method), :]))
      push!(lines, [
        k == 1 && iElem == 1 ? MultiRow(m * length(elemArr), :c, latexstring("p = $(pArr[ip])")) : "",
        k == 1 ? MultiRow(m, :c, latexstring("\\Delta_{$(row),$(col)} = $(v == 0 ? 0 : round(v, digits=3))")) : "",
        method,
        round(Int, r.coverage * 1000),
        round(r.lenCI, digits=3),
        round(r.bias * 1000, digits=1),
      ])
    end
  end
  push!(lines, Rule(:bottom))
  latex_tabular(filename, Tabular("lllccc"), lines)
end
