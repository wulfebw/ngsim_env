using AutoEnvs

function perf_ngsim_env_step()
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0750am-0805am.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
    )
    env = NGSIMEnv(params)
    n_steps = 20000
    action = [1.,0.]
    reset(env)
    @time for _ in 1:n_steps
        _, _, terminal, _ = step(env, action)
        if terminal
            reset(env)
        end
    end
end

perf_ngsim_env_step()
