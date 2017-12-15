using Base.Test
using AutoEnvs

function test_simple_ctor()
    srand(2)
    filepath = Pkg.dir("NGSIM", "data", "simple.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
        "H"=>40,
        "primesteps"=>5
    )
    env = NGSIMEnv(params)
    x = reset(env)
    nx, r, terminal, _ = step(env, [0.,0.])
    println(terminal)
    nx, r, terminal, _ = step(env, [0.,0.])
    println(terminal)
end

function test_basics()
    # ctor
    filepath = Pkg.dir("NGSIM", "data", "trajdata_debug_reduced.txt")
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

    # test infos 
    reset(env)
    _, _, _, infos = step(env, [1.,1.])
    @test infos["rmse"] != 0.
end

function test_render()
    srand(2)
    filepath = Pkg.dir("NGSIM", "data", "simple.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
        "H"=>40,
        "primesteps"=>5
    )
    env = NGSIMEnv(params)

    x = reset(env)
    imgs = []
    for _ in 1:100
        a = [1.,0.]
        img = render(env)
        x, r, terminal, _ = step(env, a)
        if terminal
            break
        end
    end
end

# @time test_simple_ctor()
# @time test_basics()
# manually test rendering
@time test_render() 
