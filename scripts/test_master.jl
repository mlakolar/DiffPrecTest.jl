# A dedicated test environment keeps extra-only master dependencies reachable on
# Julia 1.10 and leaves the package Project.toml unchanged for Aqua checks.
using Pkg, TOML
include(joinpath(@__DIR__, "master_dependencies.jl"))
package_root = dirname(@__DIR__)
project_path = joinpath(package_root, "Project.toml")
original_project = read(project_path, String)
project = TOML.parse(original_project)
deps = copy(get(project, "deps", Dict{String,Any}()))
extras = merge(get(project, "weakdeps", Dict()), get(project, "extras", Dict()))
for name in get(get(project, "targets", Dict()), "test", String[])
    deps[name] = extras[name]
end
deps[project["name"]] = project["uuid"]
compat = Dict(name => value for (name, value) in get(project, "compat", Dict())
              if name == "julia" || haskey(deps, name))
sources = Dict{String,Any}()
for name in keys(deps)
    if name == project["name"]
        sources[name] = Dict("path" => package_root)
    elseif haskey(WORKSPACE_PACKAGES, name)
        folder, _ = WORKSPACE_PACKAGES[name]
        sources[name] = Dict("path" => joinpath(dirname(package_root), folder))
    elseif haskey(get(project, "sources", Dict()), name)
        source = copy(project["sources"][name])
        haskey(source, "path") && (source["path"] = abspath(joinpath(package_root, source["path"])))
        sources[name] = source
    end
end
environment = joinpath(package_root, ".master-test-env", "v$(VERSION.major).$(VERSION.minor)")
mkpath(environment)
open(joinpath(environment, "Project.toml"), "w") do io
    TOML.print(io, Dict("deps" => deps, "compat" => compat, "sources" => sources))
end
instantiate_master(environment; workspace=dirname(package_root), owner=package_root)
read(project_path, String) == original_project || error("test setup changed package Project.toml")
installed = Pkg.dependencies()
for (name, uuid) in deps
    if name == project["name"] || haskey(WORKSPACE_PACKAGES, name)
        expected = name == project["name"] ? package_root :
                   joinpath(dirname(package_root), WORKSPACE_PACKAGES[name][1])
        realpath(installed[Base.UUID(uuid)].source) == realpath(expected) ||
            error("test environment did not retain the master checkout of $name")
    end
end
println("Master test environment: ", environment)
if !("--setup-only" in ARGS)
    depwarn = get(ENV, "JULIA_TEST_DEPWARN", "yes")
    depwarn in ("yes", "no", "error") || error("invalid JULIA_TEST_DEPWARN")
    coverage = get(ENV, "JULIA_TEST_COVERAGE", "false") == "true" ? "user" : "none"
    test_file = joinpath(package_root, "test", "runtests.jl")
    run(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --depwarn=$depwarn --code-coverage=$coverage --project=$environment $test_file`)
end
