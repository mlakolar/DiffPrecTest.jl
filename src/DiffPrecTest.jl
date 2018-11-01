module DiffPrecTest

using Statistics, StatsBase, LinearAlgebra

export
  DiffPrecResultBoot,
  DiffPrecResultNormal,
  AsymmetricOracleBoot,
  SymmetricOracleBoot,
  AsymmetricOracleNormal,
  SymmetricOracleNormal,
  estimate

# compute kron(A, B)[ind, ind]
function kron_sub!(out, A, B, ind)
  @assert size(out, 1) == size(out, 2) == length(ind)

  m, n = size(A)
  p, q = size(B)
  for col=1:length(ind)
    j = ind[col]
    ac = div(j-1, q) + 1
    bc = mod(j-1, q) + 1

    for row=1:length(ind)
        i = ind[row]

        ar = div(i-1, p) + 1
        br = mod(i-1, p) + 1

        out[row, col] = A[ar, ac] * B[br, bc]
    end
  end
  out
end

### different solvers

abstract type DiffPrecMethod end
struct AsymmetricOracleBoot <: DiffPrecMethod end
struct SymmetricOracleBoot <: DiffPrecMethod end
struct AsymmetricOracleNormal <: DiffPrecMethod end
struct SymmetricOracleNormal <: DiffPrecMethod end

###

struct DiffPrecResultBoot
  p::Float64
  boot_p::Vector{Float64}
end

struct DiffPrecResultNormal
  p::Float64
  std::Float64
end

###

# indS --- coordinates for the nonzero element of the true Δ (LinearIndices)
#          the first coordinate of indS is the one we make inference for
function estimate(::AsymmetricOracleBoot, X, Y, indS; bootSamples::Int64=1000)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = cov(X)
  Sy = cov(Y)

  A = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  Δab = (A \ (Sy[indS] - Sx[indS]))[1]

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = sample(1:nx, nx)
     bY_ind = sample(1:ny, ny)
     bSx = cov(X[bX_ind, :])
     bSy = cov(Y[bY_ind, :])

     kron_sub!(A, bSy, bSx, indS)
     boot_est[b] = (A \ (bSy[indS] - bSx[indS]))[1]
  end

  DiffPrecResultBoot(Δab, boot_est)
end


# indS --- coordinates for the nonzero element of the true Δ (LinearIndices)
function estimate(::SymmetricOracleBoot, X, Y, indS; bootSamples::Int64=1000)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = cov(X)
  Sy = cov(Y)

  # _indS = copy(indS)
  # i = LinearIndices((1:px, 1:px))[row, col]
  # if !(i in _indS)
  #     _indS = [i; _indS]
  # end
  # i = LinearIndices((1:px, 1:px))[col, row]
  # if !(i in _indS)
  #     _indS = [i; _indS]
  # end
  # j = findfirst(isequal(i), indS)

  A = zeros(length(indS), length(indS))
  B = zeros(length(indS), length(indS))
  C = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  kron_sub!(B, Sx, Sy, indS)
  @. C = (A + B) / 2.
  Δab = (C \ (Sy[indS] - Sx[indS]))[1]

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = sample(1:nx, nx)
     bY_ind = sample(1:ny, ny)
     bSx = cov(X[bX_ind, :])
     bSy = cov(Y[bY_ind, :])

     kron_sub!(A, bSy, bSx, indS)
     kron_sub!(B, bSx, bSy, indS)
     @. C = (A + B) / 2.

     boot_est[b] = (C \ (bSy[indS] - bSx[indS]))[1]
  end

  DiffPrecResultBoot(Δab, boot_est)
end

function estimate(::AsymmetricOracleNormal, X, Y, indS)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = cov(X)
  Sy = cov(Y)

  A = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  Δab = (A \ (Sy[indS] - Sx[indS]))[1]

  DiffPrecResultNormal(Δab, 0.)
end


function estimate(::SymmetricOracleNormal, X, Y, indS)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = cov(X)
  Sy = cov(Y)

  A = zeros(length(indS), length(indS))
  B = zeros(length(indS), length(indS))
  C = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  kron_sub!(B, Sx, Sy, indS)
  @. C = (A + B) / 2.
  Δab = (C \ (Sy[indS] - Sx[indS]))[1]

  A = zeros(length(indS), length(indS))
  B = zeros(length(indS), length(indS))
  kron_sub!(A, Sx, Sx, indS)
  kron_sub!(B, Sy, Sy, indS)

  DiffPrecResultNormal(Δab, (inv(A) + inv(B))[1] / nx)
end



end
