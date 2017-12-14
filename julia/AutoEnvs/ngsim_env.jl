export 
    NGSIMEnv,
    reset,
    step,
    observation_space_spec,
    action_space_spec,
    obs_names,
    render

#=
Description:
    NGSIM env that plays NGSIM trajectories, allowing the agent to take the place 
    of one of the vehicles in the trajectory
=#
type NGSIMEnv <: Env
    trajdatas::Vector{ListRecord}
    trajinfos::Vector{Dict}
    roadway::Union{Void, Roadway} # current roadway
    scene::Scene
    rec::SceneRecord
    ext::MultiFeatureExtractor
    egoid::Int # current id of relevant ego vehicle
    ego_veh::Union{Void, Vehicle} # the ego vehicle
    traj_idx::Int # current index into trajdatas 
    t::Int # current timestep in the trajdata
    h::Int # current maximum horizon for egoid
    primesteps::Int # timesteps to prime the scene
    Δt::Float64

    # metadata
    epid::Int # episode id
    render_params::Dict # rendering options
    function NGSIMEnv(
            params::Dict; 
            reclength::Int = 10,
            Δt::Float64 = .1,
            primesteps::Int = 50,
            render_params::Dict = Dict("zoom"=>5., "viz_dir"=>"/tmp"))
        param_keys = keys(params)
        @assert in("trajectory_filepaths", param_keys)
        trajdatas, trajinfos = load_ngsim_trajdatas(params["trajectory_filepaths"])
        scene_length = max_n_objects(trajdatas)
        scene = Scene(scene_length)
        rec = SceneRecord(reclength, Δt, scene_length)
        ext = build_feature_extractor(params)
        if in("render_params", param_keys)
            render_params = params["render_params"]
        end
        return new(
            trajdatas, 
            trajinfos, 
            nothing,
            scene, 
            rec, 
            ext, 
            0, nothing, 0, 0, 0, primesteps, Δt,
            0, render_params
        )
    end
end
function reset(env::NGSIMEnv)
    env.epid += 1
    empty!(env.rec)
    empty!(env.scene)
    env.traj_idx, env.egoid, env.t, env.h = sample_trajdata_vehicle(env.trajinfos)
    # prime 
    for t in env.t:(env.t + env.primesteps)
        update!(env.rec, get!(env.scene, env.trajdatas[env.traj_idx], t))
    end
    # set the ego vehicle
    env.ego_veh = env.scene[findfirst(env.scene, env.egoid)]
    # set the roadway
    env.roadway = get_corresponding_roadway(env.traj_idx)
    # env.t is the next timestep to load
    env.t += env.primesteps + 1
    return get_features(env)
end 

#=
Description:
    Propagate a single vehicle through an otherwise predeterined trajdata

Args:
    - env: environment to be stepped forward
    - action: array of floats that can be converted into an AccelTurnrate
=#
function _step!(env::NGSIMEnv, action::Array{Float64})
    # convert action into form 
    ego_action = AccelTurnrate(action...)
    # propagate the ego vehicle 
    ego_state = propagate(
        env.ego_veh, 
        ego_action, 
        env.roadway, 
        env.Δt
    )
    # update the ego_veh
    env.ego_veh = Entity(env.ego_veh, ego_state)

    # load the actual scene, and insert the vehicle into it
    get!(env.scene, env.trajdatas[env.traj_idx], env.t)
    env.scene[findfirst(env.scene, env.egoid)] = env.ego_veh

    # update rec with current scene 
    update!(env.rec, env.scene)
end
function Base.step(env::NGSIMEnv, action::Array{Float64})
    _step!(env, action)
    # update env timestep to be the next scene to load
    env.t += 1
    terminal = env.t >= env.h ? true : false
    return get_features(env), 0, terminal, Dict()
end
function AutoRisk.get_features(env::NGSIMEnv)
    veh_idx = findfirst(env.scene, env.egoid)
    pull_features!(env.ext, env.rec, env.roadway, veh_idx)
    return deepcopy(env.ext.features)
end
function observation_space_spec(env::NGSIMEnv)
    low = zeros(length(env.ext))
    high = zeros(length(env.ext))
    feature_infos = feature_info(env.ext)
    for (i, fn) in enumerate(feature_names(env.ext))
        low[i] = feature_infos[fn]["low"]
        high[i] = feature_infos[fn]["high"]
    end
    infos = Dict("high"=>high, "low"=>low)
    return (length(env.ext),), "Box", infos
end
action_space_spec(env::NGSIMEnv) = (2,), "Box", Dict("high"=>[1.,1.], "low"=>[-1.,-1.])
obs_names(env::NGSIMEnv) = feature_names(env.ext)

#=
Description:
    Render the scene 

Args:
    - env: environment to render

Returns:
    - img: returns a (height, width, channel) image to display
=#
function render(env::NGSIMEnv)
    # define colors for all the vehicles
    carcolors = Dict{Int,Colorant}()
    for veh in env.scene
        carcolors[veh.id] = veh.id == env.egoid ? colorant"red" : colorant"green"
    end

    # define a camera following the ego vehicle
    cam = AutoViz.CarFollowCamera{Int}(env.egoid, env.render_params["zoom"])
    stats = [
        CarFollowingStatsOverlay(env.egoid, 2), 
        NeighborsOverlay(env.egoid, textparams = TextParams(x = 600, y_start=300))
    ]

    # render the frame
    frame = render(
        env.scene, 
        env.roadway,
        stats, 
        cam = cam, 
        car_colors = carcolors
    )

    # save the frame 
    if !isdir(env.render_params["viz_dir"])
        mkdir(env.render_params["viz_dir"])
    end
    ep_dir = joinpath(env.render_params["viz_dir"], "episode_$(env.epid)")
    if !isdir(ep_dir)
        mkdir(ep_dir)
    end
    filepath = joinpath(ep_dir, "step_$(env.t).png")
    write_to_png(frame, filepath)

    # load and return the frame as an rgb array
    img = PyPlot.imread(filepath)
    return img
end

