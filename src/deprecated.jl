# Compatibility spellings for the naming migration. Binding aliases retain
# generic-function identity so downstream method extensions keep dispatching.
Base.@deprecate_binding computeSimulationResult compute_simulation_result
Base.@deprecate_binding diffEstimation diff_estimation
Base.@deprecate_binding invAsymHessianEstimation inv_asym_hessian_estimation
Base.@deprecate_binding invQSymHessian inv_q_sym_hessian
Base.@deprecate_binding invHessianReduced inv_hessian_reduced
Base.@deprecate_binding supportEstimate support_estimate
Base.@deprecate_binding simulCI simul_ci
Base.@deprecate_binding simulCIstudentized simul_ci_studentized
Base.@deprecate_binding getSupport get_support false
Base.@deprecate_binding getReducedSupport get_reduced_support false
Base.@deprecate_binding getLinearSupport get_linear_support false
Base.@deprecate_binding makeFirst! make_first! false
Base.@deprecate_binding createVarianceStatistic create_variance_statistic false
Base.@deprecate_binding createCovarianceStatistic create_covariance_statistic false
Base.@deprecate_binding createSecondMomentStatistic create_second_moment_statistic false
Base.@deprecate_binding reducedDiffEstimation reduced_diff_estimation false

@inline function Base.getproperty(result::SimulationResult, name::Symbol)
  if name === :lenCoverage
    Base.depwarn("property lenCoverage is deprecated; use len_coverage.", :lenCoverage)
    return getfield(result, :len_coverage)
  end
  getfield(result, name)
end
Base.propertynames(::SimulationResult, private::Bool=false) =
  (fieldnames(SimulationResult)..., :lenCoverage)
