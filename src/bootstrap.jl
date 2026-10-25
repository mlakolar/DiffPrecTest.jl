"""
    BootstrapEstimates

Refitted estimates `Δhat` of `svec(Δ)`, the lower triangle of `Δ` in the order
of `ProximalBase.tril_index` with the off-diagonal entries multiplied
by `√2`, and their bootstrap replicates `Δb`, one replicate per column.
"""
struct BootstrapEstimates
  Δhat::Vector{Float64}
  Δb::Matrix{Float64}     # each column represents a bootstrap estimate
end


"""
    bootstrap(X, Y; boot_samples=300, estim_support=nothing, rng=Random.default_rng())

Refit every entry of `svec(Δ)` on its estimated support and compute
`boot_samples` bootstrap replicates of the refitted estimates, resampling the
rows of `X` and `Y`. Returns `(straps, e_s)`: the [`BootstrapEstimates`](@ref)
and the supports, as vectors of indices into `svec(Δ)`, one per entry. The
supports are computed by the first and second stage estimators of
[`ReducedNormal`](@ref) unless given in `estim_support`; the entry itself is
moved to the front of each support, which is modified.

Use [`simul_ci`](@ref) or [`simul_ci_studentized`](@ref) for simultaneous
confidence intervals.
"""
function bootstrap(X, Y; boot_samples=nothing, bootSamples=nothing, estim_support=nothing, estimSupport=nothing,
                   rng::AbstractRNG=Random.default_rng())
  boot_samples = _deprecated_keyword(boot_samples, bootSamples, :boot_samples, :bootSamples, 300)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
  nx, p = size(X)
  ny    = size(Y, 1)

  Sx = Symmetric( cov(X) )
  Sy = Symmetric( cov(Y) )

  e_s = estim_support === nothing ? __init_support(Sx, Sy, X, Y) : estim_support

  rp = div((p+1)*p, 2)
  # ind_s = Vector{Vector{Int64}}(undef, rp)

  Δhat = Vector{Float64}(undef, rp)
  Δb = Matrix{Float64}(undef, rp, boot_samples)

  b_x = similar(X)
  b_y = similar(Y)
  x_ind = Vector{Int64}(undef, nx)
  y_ind = Vector{Int64}(undef, ny)

  # obtain initial estimate
  it = 0
  for col=1:p
    for row=col:p
      it = it + 1
      ind_elem = tril_index(p, row, col)
      # tmp = estimate(ReducedOracleNormal(), Sx, Sy, X, Y, ind_elem, e_s[it])

      ind_s = e_s[it]
      make_first!(ind_s, ind_elem)
      C = skron(Sx, Sy, ind_s)
      b = svec(Sy, ind_s) - svec(Sx, ind_s)
      Δ = C \ b

      Δhat[it] = Δ[1]
    end
  end

  for _b=1:boot_samples
    rand!(rng, x_ind, 1:nx)
    rand!(rng, y_ind, 1:ny)

    _fill_boot!(b_x, X, x_ind)
    _fill_boot!(b_y, Y, y_ind)

    b_sx = Symmetric( cov(b_x) )
    b_sy = Symmetric( cov(b_y) )

    it = 0
    for col=1:p
      for row=col:p
        it = it + 1
        ind_elem = tril_index(p, row, col)
        # tmp = estimate(ReducedOracleNormal(), b_sx, b_sy, b_x, b_y, index_element_linear, e_s[it])

        ind_s = e_s[it]
        make_first!(ind_s, ind_elem)
        C = skron(b_sx, b_sy, ind_s)
        b = svec(b_sy, ind_s) - svec(b_sx, ind_s)
        Δ = C \ b

        Δb[it, _b] = Δ[1]
      end
    end
  end

  BootstrapEstimates(Δhat, Δb), e_s
end





"""
    simul_ci(straps::BootstrapEstimates, α=0.95)

Simultaneous confidence intervals with coverage `α` for the entries of
`svec(Δ)`, `Δhat .± x` with `x` the `α` quantile of the bootstrap distribution
of `‖Δhat - Δb‖∞`. Returns a matrix with the lower ends of the intervals in
the first column and the upper ends in the second.
"""
function simul_ci(straps::BootstrapEstimates, α::Float64=0.95)
    m, boot_samples = size(straps.Δb)

    inf_norm_dist = Vector{Float64}(undef, boot_samples)
    CI = Matrix{Float64}(undef, m, 2)

    for b=1:boot_samples
        inf_norm_dist[b] = norm_diff(straps.Δhat, view(straps.Δb, :, b), Inf)
    end
    x = quantile!(inf_norm_dist, α)
    CI[:, 1] .= straps.Δhat .- x
    CI[:, 2] .= straps.Δhat .+ x

    CI
end

"""
    simul_ci_studentized(straps::BootstrapEstimates, α=0.95)

Like [`simul_ci`](@ref), but every entry is standardized by its bootstrap
standard error `w`: the intervals are `Δhat .± x .* w` with `x` the `α` quantile
of the bootstrap distribution of `‖(Δhat - Δb) ./ w‖∞`.
"""
function simul_ci_studentized(straps::BootstrapEstimates, α::Float64=0.95)
    m, boot_samples = size(straps.Δb)

    inf_norm_dist = Vector{Float64}(undef, boot_samples)
    w = reshape(std(straps.Δb; dims = 2, corrected = false), :)

    CI = Matrix{Float64}(undef, m, 2)
    tmp = Vector{Float64}(undef, m)

    for b=1:boot_samples
        tmp .= (straps.Δhat .- straps.Δb[:, b]) ./ w
        inf_norm_dist[b] = norm(tmp, Inf)
    end
    x = quantile!(inf_norm_dist, α)
    @. CI[:, 1] = straps.Δhat - x * w
    @. CI[:, 2] = straps.Δhat + x * w

    CI
end


function _fill_boot!(b_x, X, b_ind)
  n, p = size(X)
  for col=1:p
      for row=1:n
          b_x[row, col] = X[b_ind[row], col]
      end
  end
end
