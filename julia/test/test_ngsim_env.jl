using Base.Test
using AutoEnvs

function test_basics()
    # ctor
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0750am-0805am.txt")
    params = Dict("trajectory_filepaths"=>[filepath])
    env = NGSIMEnv(params)

    # reset, step
    x = reset(env)
    a = [0., 0.]
    nx, r, terminal, infos = step(env, a)
    @test x != nx
    @test terminal == false
    nnx, r, terminal, infos = step(env, a)
    @test nx != nnx

    # obs spec
    shape, spacetype, infos = observation_space_spec(env)
    @test spacetype == "Box"
    @test in("high", keys(infos))
    @test in("low", keys(infos))
    
    # does accel reflect applied value
    fns = obs_names(env)
    acc_idx = [i for (i,n) in enumerate(fns) if "accel" == n][1]
    tur_idx = [i for (i,n) in enumerate(fns) if "turn_rate_global" == n][1]
    a = [0., 1.]
    for _ in 1:10
        nx, _, _, _ = step(env, a)
    end
    
    @test abs(nx[acc_idx]) <= 1e-1
    @test abs(nx[tur_idx] - 1) <= 1e-1
end

function test_render()
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0750am-0805am.txt")
    params = Dict("trajectory_filepaths"=>[filepath])
    env = NGSIMEnv(params)

    x = reset(env)
    imgs = []
    for _ in 1:100
        a = [1.,1.]
        img = render(env)
        nx, r, terminal, _ = step(env, a)
    end
end

@time test_basics()
# manually test rendering
# @time test_render() 
