
include("invHessianEstimationASym.jl")
include("invHessianEstimationSymReduced.jl")


########################
#
# util functions
#
########################
function _mul(S::Symmetric, θ::SparseIterate)
    p = size(S, 1)

    out = zeros(p, p)
    for ci=1:p
        for ri=1:p
            for k=1:p
                @inbounds out[ri, ci] += S[ri, k] * θ[(ci-1)*p + k]
            end
        end
    end
    out
end

function _mul(θ::SparseIterate, S::Symmetric)
    p = size(S, 1)

    out = zeros(p, p)
    for ci=1:p
        for ri=1:p
            for k=1:p
                @inbounds out[ri, ci] += θ[(k-1)*p + ri] * S[k, ci]
            end
        end
    end
    out
end
