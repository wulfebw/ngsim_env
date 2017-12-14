using Base.Test
using AutoEnvs

function runtests()
    include("test_debug_envs.jl")
    include("test_ngsim_utils.jl")
    include("test_ngsim_env.jl")
end

@time runtests()
