# Development conventions

Types and modules use UpperCamelCase. Descriptive functions, fields and keyword
arguments use snake_case; mutating functions end in `!`. Retain standard numerical
abbreviations, mathematical identifiers and names required by upstream interfaces.
Run `python3 scripts/check_naming.py` to check new declarations.

Workspace dependencies use sibling master checkouts, including transitive
dependencies. Missing checkouts are cloned from master. Existing checkouts are
never reset or silently updated. Package version numbering and registry changes
are deferred.

Run the complete tests from the repository root:

```sh
julia scripts/test_master.jl
```

The script prepares an isolated master-based environment with test extras on
Julia 1.10 and newer, then runs the suite without changing package dependencies.
For the simulation environment, use `julia scripts/instantiate.jl exper`.
