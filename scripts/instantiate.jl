# Resolve every workspace dependency from a local master checkout.
# Missing sibling checkouts are cloned from their master branches.
include(joinpath(@__DIR__, "master_dependencies.jl"))
package_root = dirname(@__DIR__)
environment = isempty(ARGS) ? package_root : abspath(ARGS[1])
instantiate_master(environment; workspace=dirname(package_root), owner=package_root)
