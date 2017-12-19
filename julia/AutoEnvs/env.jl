export 
    Env,
    step,
    reset,
    observation_space_spec,
    action_space_spec,
    render,
    obs_names,
    reward_names

abstract type Env end
Base.step(env::Env, action::Int) = error("Not implemented")
Base.step(env::Env, action::Float64) = error("Not implemented")
Base.step(env::Env, action::Array{Float64}) = error("Not implemented")
Base.reset(env::Env) = error("Not implemented")
observation_space_spec(env::Env) = error("Not implemented")
action_space_spec(env::Env) = error("Not implemented")
render(env::Env) = error("Not implemented")
obs_names(env::Env) = error("Not implemented")