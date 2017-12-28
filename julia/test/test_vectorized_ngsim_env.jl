using Base.Test
using AutoEnvs

function test_basics()
    # ctor
    filepath = Pkg.dir("NGSIM", "data", "trajdata_debug_reduced.txt")
    n_envs = 100
    params = Dict("trajectory_filepaths"=>[filepath], "n_envs"=>n_envs)
    env = VectorizedNGSIMEnv(params)

    # reset, step
    x = reset(env)
    a = zeros(n_envs, 2)
    nx, r, terminal, infos = step(env, a)
    @test x != nx
    nnx, r, terminal, infos = step(env, a)
    @test nx != nnx

    # obs spec
    shape, spacetype, infos = observation_space_spec(env)
    @test spacetype == "Box"
    @test in("high", keys(infos))
    @test in("low", keys(infos))
    
end

@time test_basics()
