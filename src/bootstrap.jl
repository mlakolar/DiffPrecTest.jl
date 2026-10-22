"""
    BootstrapEstimates

Refitted estimates `Δhat` of `svec(Δ)`, the lower triangle of `Δ` in the order
of `ProximalBase.sub2indLowerTriangular` with the off-diagonal entries multiplied
by `√2`, and their bootstrap replicates `Δb`, one replicate per column.
"""
struct BootstrapEstimates
  Δhat::Vector{Float64}
  Δb::Matrix{Float64}     # each column represents a bootstrap estimate
end


"""
    bootstrap(X, Y; bootSamples=300, estimSupport=nothing, rng=Random.default_rng())

Refit every entry of `svec(Δ)` on its estimated support and compute
`bootSamples` bootstrap replicates of the refitted estimates, resampling the
rows of `X` and `Y`. Returns `(straps, eS)`: the [`BootstrapEstimates`](@ref)
and the supports, as vectors of indices into `svec(Δ)`, one per entry. The
supports are computed by the first and second stage estimators of
[`ReducedNormal`](@ref) unless given in `estimSupport`; the entry itself is
moved to the front of each support, which is modified.

Use [`simulCI`](@ref) or [`simulCIstudentized`](@ref) for simultaneous
confidence intervals.
"""
function bootstrap(X, Y; bootSamples::Int64=300, estimSupport::Union{Vector{Vector{Int64}},Nothing}=nothing,
                   rng::AbstractRNG=Random.default_rng())
  nx, p = size(X)
  ny    = size(Y, 1)

  Sx = Symmetric( cov(X) )
  Sy = Symmetric( cov(Y) )

  eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport

  rp = div((p+1)*p, 2)
  # indS = Vector{Vector{Int64}}(undef, rp)

  Δhat = Vector{Float64}(undef, rp)
  Δb = Matrix{Float64}(undef, rp, bootSamples)

  bX = similar(X)
  bY = similar(Y)
  x_ind = Vector{Int64}(undef, nx)
  y_ind = Vector{Int64}(undef, ny)

  # obtain initial estimate
  it = 0
  for col=1:p
    for row=col:p
      it = it + 1
      indElem = sub2indLowerTriangular(p, row, col)
      # tmp = estimate(ReducedOracleNormal(), Sx, Sy, X, Y, indElem, eS[it])

      indS = eS[it]
      makeFirst!(indS, indElem)
      C = skron(Sx, Sy, indS)
      b = svec(Sy, indS) - svec(Sx, indS)      
      Δ = C \ b

      Δhat[it] = Δ[1]
    end
  end

  for _b=1:bootSamples
    rand!(rng, x_ind, 1:nx)
    rand!(rng, y_ind, 1:ny)

    _fill_boot!(bX, X, x_ind)
    _fill_boot!(bY, Y, y_ind)

    bSx = Symmetric( cov(bX) )
    bSy = Symmetric( cov(bY) )

    it = 0
    for col=1:p
      for row=col:p
        it = it + 1
        indElem = sub2indLowerTriangular(p, row, col)
        # tmp = estimate(ReducedOracleNormal(), bSx, bSy, bX, bY, indexElementLinear, eS[it])

        indS = eS[it]
        makeFirst!(indS, indElem)
        C = skron(bSx, bSy, indS)
        b = svec(bSy, indS) - svec(bSx, indS)      
        Δ = C \ b

        Δb[it, _b] = Δ[1]
      end
    end
  end

  BootstrapEstimates(Δhat, Δb), eS
end





"""
    simulCI(straps::BootstrapEstimates, α=0.95)

Simultaneous confidence intervals with coverage `α` for the entries of
`svec(Δ)`, `Δhat .± x` with `x` the `α` quantile of the bootstrap distribution
of `‖Δhat - Δb‖∞`. Returns a matrix with the lower ends of the intervals in
the first column and the upper ends in the second.
"""
function simulCI(straps::BootstrapEstimates, α::Float64=0.95)
    m, bootSamples = size(straps.Δb)

    infNormDist = Vector{Float64}(undef, bootSamples)
    CI = Matrix{Float64}(undef, m, 2)

    for b=1:bootSamples
        infNormDist[b] = norm_diff(straps.Δhat, view(straps.Δb, :, b), Inf)
    end
    x = quantile!(infNormDist, α)
    CI[:, 1] .= straps.Δhat .- x
    CI[:, 2] .= straps.Δhat .+ x

    CI
end

"""
    simulCIstudentized(straps::BootstrapEstimates, α=0.95)

Like [`simulCI`](@ref), but every entry is standardized by its bootstrap
standard error `w`: the intervals are `Δhat .± x .* w` with `x` the `α` quantile
of the bootstrap distribution of `‖(Δhat - Δb) ./ w‖∞`.
"""
function simulCIstudentized(straps::BootstrapEstimates, α::Float64=0.95)
    m, bootSamples = size(straps.Δb)

    infNormDist = Vector{Float64}(undef, bootSamples)
    w = reshape(std(straps.Δb; dims = 2, corrected = false), :)

    CI = Matrix{Float64}(undef, m, 2)
    tmp = Vector{Float64}(undef, m)

    for b=1:bootSamples
        tmp .= (straps.Δhat .- straps.Δb[:, b]) ./ w
        infNormDist[b] = norm(tmp, Inf)
    end
    x = quantile!(infNormDist, α)
    @. CI[:, 1] = straps.Δhat - x * w
    @. CI[:, 2] = straps.Δhat + x * w

    CI
end


function _fill_boot!(bX, X, b_ind)
  n, p = size(X)
  for col=1:p
      for row=1:n
          bX[row, col] = X[b_ind[row], col]
      end
  end
end
