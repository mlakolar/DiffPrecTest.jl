struct BootstrapEstimates
    Δhat::Vector{Float64}
    Δb::Matrix{Float64}     # each column represents a bootstrap estimate
end


function bootstrap(X, Y; bootSamples::Int64=300, estimSupport::Union{Array{BitArray},Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport

    rp = div((p+1)*p, 2)
    indS = Vector{Vector{Int64}}(undef, rp)

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
             indS[it] = getLinearSupport(row, col, eS[it])

             tmp = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS[it])
             Δhat[it] = tmp.p
         end
     end

    for b=1:bootSamples
        sample!(1:nx, x_ind)
        sample!(1:ny, y_ind)

        _fill_boot!(bX, X, x_ind)
        _fill_boot!(bY, Y, y_ind)

        bSx = Symmetric( cov(bX) )
        bSy = Symmetric( cov(bY) )

        it = 0
        for col=1:p
            for row=col:p
                it = it + 1

                tmp = estimate(SymmetricOracleNormal(), bSx, bSy, bX, bY, row, col, indS[it])
                Δb[it, b] = tmp.p
            end
        end
    end

    BootstrapEstimates(Δhat, Δb), eS
end





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
