
###############
#
# Oracle
#
###############

function estimate(
  ::ReducedOracleNormal,
  Sx::Symmetric,
  Sy::Symmetric,
  X,
  Y,
  indElem,  
  indS::Vector{Int64}
  )

  makeFirst!(indS, indElem)
  C = skron(Sx, Sy, indS)
  b = svec(Sy, indS) - svec(Sx, indS)

  tmp = zeros(length(indS))
  tmp[1] = 1.
  ω = C \ tmp
  Δ = C \ b
  v = variance(ReducedOracleNormal(), Sx, Sy, X, Y, ω, indS, Δ, indS)

  row, col = ind2subLowerTriangular(size(Sx, 1), indElem)
  if row == col
    return DiffPrecResultNormal(Δ[1], sqrt(v))
  else
    return DiffPrecResultNormal(Δ[1] / sqrt(2.), sqrt(v / 2.))
  end
end



######################### first stage ############################################



function _computeVarStep1!(ω, X, Y, Δ::SparseIterate)
  nx = size(X, 1)
  ny = size(Y, 1)

  q = zeros(nx)
  r = zeros(ny)

  for ri=1:length(ω)
    fill!(q, 0.)
    fill!(r, 0.)

    @inbounds for k=1:nx
      vx = view(X, k, :)
      svec_x_ri = svec(vx, vx, ri)
      for j=1:ny
        vy = view(Y, j, :)
        # skron(vx_k, vy_j)[ri, :] * Δ
        out = 0.
        for inz = 1:nnz(Δ)
          indColumn = Δ.nzval2ind[inz]          
          out += skron(vx, vy, ri, indColumn) * Δ.nzval[inz]
        end
        q[k] += out + svec_x_ri - svec(vy, vy, ri)
      end
      q[k] /= ny
    end

    @inbounds for k=1:ny
      vy = view(Y, k, :)
      svec_y_ri = svec(vy, vy, ri)
      for j=1:nx
        vx = view(X, j, :)
        # skron(vx_j, vy_k)[ri, :] * Δ
        out = 0.
        for inz = 1:nnz(Δ)
          indColumn = Δ.nzval2ind[inz]          
          out += skron(vx, vy, ri, indColumn) * Δ.nzval[inz]
        end
        r[k] += out + svec(vx, vx, ri) - svec_y_ri
      end
      r[k] /= nx
    end

    t = mean(q)

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2
  
    ω[ri] = sqrt( σ1/nx + σ2/ny )
  end
end





function reducedDiffEstimation(H::Symmetric, b::Vector, X, Y, λ, options=CDOptions())
  f = CDQuadraticLoss(H, b)
  x = SparseIterate(size(H, 1))

  ##################
  #
  #  first stage
  #
  ##################
  # compute initial variance
  ω = Array{eltype(H)}(undef, length(x))
  _computeVarStep1!(ω, X, Y, x)

  # compute initial estimate
  g = ProxL1(λ, ω)
  coordinateDescent!(x, f, g, options)

  ##################
  #
  #  second stage
  #
  ##################

  # recompute variance
  opt1 = CDOptions(;
      maxIter=options.maxIter,
      optTol=options.optTol,
      randomize=options.randomize,
      warmStart=true,
      numSteps=options.numSteps)

  _computeVarStep1!(ω, X, Y, x)

  # recompute estimate
  g = ProxL1(λ, ω)
  coordinateDescent!(x, f, g, opt1)

  return x
end


######################### second stage ############################################



function _computeVarStep2!(ω, indElem, X, Y, Δ::SparseIterate)
  nx = size(X, 1)
  ny = size(Y, 1)

  q = zeros(nx)
  r = zeros(ny)

  for ri=1:length(ω)
    fill!(q, 0.)
    fill!(r, 0.)

    δ = indElem == ri ? - 1. : 0.

    @inbounds for k=1:nx
      vx = view(X, k, :)      
      for j=1:ny
        vy = view(Y, j, :)
        # skron(vx_k, vy_j)[ri, :] * Δ
        out = 0.
        for inz = 1:nnz(Δ)
          indColumn = Δ.nzval2ind[inz]          
          out += skron(vx, vy, ri, indColumn) * Δ.nzval[inz]
        end
        q[k] += out + δ
      end
      q[k] /= ny
    end

    @inbounds for k=1:ny
      vy = view(Y, k, :)      
      for j=1:nx
        vx = view(X, j, :)
        # skron(vx_j, vy_k)[ri, :] * Δ
        out = 0.
        for inz = 1:nnz(Δ)
          indColumn = Δ.nzval2ind[inz]          
          out += skron(vx, vy, ri, indColumn) * Δ.nzval[inz]
        end
        r[k] += out + δ
      end
      r[k] /= nx
    end

    t = mean(q)

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2
  
    ω[ri] = sqrt( σ1/nx + σ2/ny )
  end
end


function invHessianReduced(H::Symmetric, indRow::Int, X, Y, λ, options=CDOptions())
  b = zeros(size(H, 1))
  b[indRow] = 1.
  f = CDQuadraticLoss(H, b)
  x = SparseIterate(size(H, 1))

  ##################
  #
  #  first stage
  #
  ##################
  # compute initial variance
  ω = Array{eltype(H)}(undef, length(x))
  _computeVarStep2!(ω, indRow, X, Y, x)

  # compute initial estimate
  g = ProxL1(λ, ω)
  coordinateDescent!(x, f, g, options)

  ##################
  #
  #  second stage
  #
  ##################

  # recompute variance
  opt1 = CDOptions(;
      maxIter=options.maxIter,
      optTol=options.optTol,
      randomize=options.randomize,
      warmStart=true,
      numSteps=options.numSteps)

  _computeVarStep2!(ω, indRow, X, Y, x)

  # recompute estimate
  g = ProxL1(λ, ω)
  coordinateDescent!(x, f, g, opt1)

  return x
end

########################################################################

# indElem --- index in a lower triangular part of the matrix
#             can be obtained by sub2indLowerTriangular        
function estimate(::ReducedNormal, X, Y, indElem;
  Sx::Union{Symmetric, Nothing}=nothing,
  Sy::Union{Symmetric, Nothing}=nothing,
  H::Union{Symmetric, Nothing}=nothing,
  Δ::Union{SymmetricSparseIterate, Nothing}=nothing,
  suppΔ::Union{BitArray{1}, Nothing}=nothing,
  ω::Union{SparseIterate, Nothing}=nothing,
  suppω::Union{BitArray{1}, Nothing}=nothing
  )

  nx, p = size(X)
  ny    = size(Y, 1)

  if Sx === nothing
    Sx = Symmetric( X'X / nx )
  end
  if Sy === nothing
    Sy = Symmetric( Y'Y / ny )
  end
  if H === nothing
    H = Symmetric( skron(Sx, Sy) )
  end
  b = svec(Sx) - svec(Sy)

  # first stage
  λ = 1.01 * quantile( Normal(), 1. - 0.1 / (p*(p+1)) )
  if Δ === nothing && suppΔ === nothing
    Δ = reducedDiffEstimation(H, b, X, Y, λ)
    suppΔ = getReducedSupport(Δ)
  end
  if suppΔ === nothing
    suppΔ = getReducedSupport(Δ)
  end

  # second stage
  if ω === nothing && suppω === nothing    
    ω = invHessianReduced(H, indElem, X, Y, λ)
    suppω = getReducedSupport(ω)
  end
  if suppω === nothing
    suppω = getReducedSupport(ω)
  end

  # refit stage
  indS = (1:size(H,1))[suppΔ .| suppω]
  estimate(ReducedOracleNormal(), Sx, Sy, X, Y, indElem, indS), Δ, suppΔ, ω, suppω, indS
end

