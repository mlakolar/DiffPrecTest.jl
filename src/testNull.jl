

function testGlobalNull(X, Y; bootSamples::Int64=300, estimSupport::Union{Array{BitArray},Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport

    rp = div((p+1)*p, 2)
    indS = Vector{Vector{Int64}}(undef, rp)

    Δhat = Vector{Float64}(undef, rp)

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


    # this is not complete yet....
end