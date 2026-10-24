####################################
#
#  Precompilation
#
####################################

# The estimators' first calls, compiled into the package image. PrecompileTools
# runs this workload when the package is precompiled; the preference
# `precompile_workload = false` skips it.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  p, n = 5, 300
  Σy = [0.5^abs(i - j) for i in 1:p, j in 1:p]
  X = randn(Xoshiro(1), n, p)
  Y = randn(Xoshiro(2), n, p) * cholesky(Σy).U
  @compile_workload begin
    estimate(SymmetricNormal(), X, Y, 2, 1)
    estimate(AsymmetricNormal(), X, Y, 2, 1)
    estimate(SeparateNormal(), X, Y, 2, 1)
    estimate(ReducedNormal(), X, Y, tril_index(p, 2, 1))
    straps, = bootstrap(X, Y; bootSamples = 2)
    simulCIstudentized(straps, 0.95)
  end
end
