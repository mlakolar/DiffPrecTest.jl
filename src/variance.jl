

#
# output 
function symmetricVariance!(out, Sx, Sy, X, Y, Δ, grad)
  p = size(Sx, 1)
  nx = size(X, 1)
  ny = size(Y, 1)

  length(out) == div( p * (p + 1), 2 ) || throw(ArgumentError("Dimension of out wrong"))
    
  fill!(out, 0.)
  
  # Qt --- stores Sy Δ X'  (p x nx)
  Qt = zeros(p, nx)
  mul!(Qt, Δ, transpose(X))
  lmul!(Sy, Qt)  
    
end