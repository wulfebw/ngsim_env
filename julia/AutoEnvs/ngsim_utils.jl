export 
    index_ngsim_trajectory,
    load_ngsim_trajdatas,
    sample_trajdata_vehicle,
    build_feature_extractor,
    max_n_objects,
    fill_infos_cache,
    stack_tensor_dict_list

#=
Description:
    Creates an index (easily accessible set of information) for an ngsim trajdata.
    The index is specifically a dictionary mapping vehicle ids to a dictionary 
    of metadata, for example the first and last timestep where that vehicle is 
    present in the trajdata.

Args:
    - filepath: filepath to trajdata to load 
    - minlength: minimum trajectory length to include
    - vebose: print out progress

Returns:
    - index: index of the trajdata
=#
function index_ngsim_trajectory(
        filepath::String; 
        minlength::Int=100,
        verbose::Int=1)
    # setup
    index = Dict()
    trajdata = load_trajdata(filepath)
    n_frames = nframes(trajdata)
    scene_length = maximum(n_objects_in_frame(trajdata, i) for i in 1 : n_frames)
    scene = Scene(scene_length)
    prev, cur = Set(), Set()

    # iterate each frame collecting info about the vehicles
    for frame in 1 : n_frames
        if verbose > 0
            print("\rframe $(frame) / $(n_frames)")
        end
        cur = Set()
        get!(scene, trajdata, frame)

        # add all the vehicles to the current set
        for veh in scene
            push!(cur, veh.id)
            # insert previously unseen vehicles into the index
            if !in(veh.id, prev)
                index[veh.id] = Dict("ts"=>frame)
            end
        end

        # find vehicles in the previous but not the current frame
        missing = setdiff(prev, cur)
        for id in missing
            # set the final frame for all these vehicles
            index[id]["te"] = frame - 1
        end

        # step forward
        prev = cur
    end

    # at this point, any ids in cur are in the last frame, so add them in 
    for id in cur
        index[id]["te"] = n_frames
    end

    # postprocess to remove undesirable trajectories
    for (vehid, infos) in index
        # check for start and end frames 
        if !in("ts", keys(infos)) || !in("te", keys(infos))
            if verbose > 0
                println("delete vehid $(vehid) for missing keys")
            end
            delete!(index, vehid)

        # check for start and end frames greater than minlength
        elseif infos["te"] - infos["ts"] < minlength
            if verbose > 0
                println("delete vehid $(vehid) for below minlength")
            end
            delete!(index, vehid)
        end
    end

    return index
end

#=
Description:
    Loads trajdatas and metadata used for sampling individual trajectories

Args:
    - filepaths: list of filepaths to individual trajdatas

Returns:
    - trajdatas: list of trajdata objects, each to a timeperiod of NGSIM
    - trajinfos: list of dictionaries providing metadata for sampling
        each dictionary has 
            key = id of a vehicle in the trajdata
            value = first and last timestep in trajdata of vehicle
=#
function load_ngsim_trajdatas(filepaths; minlength::Int=100)
    # check that indexes exist for the relevant trajdatas
    # if they are missing, create the index
    # the index is just a collection of metadata that is saved with the 
    # trajdatas to allow for a more efficient environment implementation
    indexes_filepaths = [replace(f, ".txt", "-index.jld") for f in filepaths]
    indexes = Dict[]
    for (i, index_filepath) in enumerate(indexes_filepaths)
        # check if index already created
        # if so, load it
        # if not, create and save it
        if !isfile(index_filepath)
            index = index_ngsim_trajectory(filepaths[i], minlength=minlength)
            JLD.save(index_filepath, "index", index)
        else
            index = JLD.load(index_filepath)["index"]
        end

        # load index
        push!(indexes, index)
    end

    # load trajdatas
    trajdatas = Records.ListRecord[]
    roadways = Roadway[]
    for filepath in filepaths
        trajdata = load_trajdata(filepath)
        push!(trajdatas, trajdata)
        roadway = get_corresponding_roadway(filepath)
        push!(roadways, roadway)
    end

    return trajdatas, indexes, roadways
end

#=
Description:
    Sample a vehicle to imitate

Args:
    - trajinfos: the metadata list of dictionaries

Returns:
    - traj_idx: index of NGSIM trajdatas
    - egoid: id of ego vehicle
    - ts: start timestep for vehicle 
    - te: end timestep for vehicle
=#
function sample_trajdata_vehicle(
        trajinfos, 
        offset::Int=0,
        traj_idx::Union{Void,Int}=nothing,
        egoid::Union{Void,Int}=nothing)
    if traj_idx == nothing || egoid == nothing
        traj_idx = rand(1:length(trajinfos))
        egoid = rand(collect(keys(trajinfos[traj_idx])))
    end

    ts = trajinfos[traj_idx][egoid]["ts"]
    te = trajinfos[traj_idx][egoid]["te"]
    # sample actual start timestep from [ts, te-offset] (assume valid range)
    ts = rand(ts:te - offset)
    return traj_idx, egoid, ts, te
end

function build_feature_extractor(params = Dict())
    subexts::Vector{AbstractFeatureExtractor} = []
    push!(subexts, CoreFeatureExtractor())
    push!(subexts, TemporalFeatureExtractor())
    push!(subexts, WellBehavedFeatureExtractor())
    push!(subexts, CarLidarFeatureExtractor(20, carlidar_max_range = 50.))
    push!(subexts, ForeForeFeatureExtractor())
    ext = MultiFeatureExtractor(subexts)
    return ext
end

function max_n_objects(trajdatas)
    cur_max = -1
    for trajdata in trajdatas
        cur = maximum(n_objects_in_frame(trajdata, i) for i in 1 : nframes(trajdata))
        cur_max = max(cur, cur_max)
    end
    return cur_max
end

function fill_infos_cache(ext::MultiFeatureExtractor)
    cache = Dict()
    cache["feature_names"] = feature_names(ext)
    for (i,n) in enumerate(cache["feature_names"])
        if "is_colliding" == n
            cache["is_colliding_idx"] = i
        end
        if "out_of_lane" == n 
            cache["out_of_lane_idx"] = i
        end
        if "markerdist_left" == n
            cache["markerdist_left_idx"] = i
        end
        if "markerdist_right" == n
            cache["markerdist_right_idx"] = i
        end
    end
    return cache
end

function stack_tensor_dict_list(lst::Vector{Dict})
    dict_keys = collect(keys(lst[1]))
    ret = Dict()
    for k in dict_keys
        example = lst[1][k]
        if isa(example, Dict)
            v = stack_tensor_dict_list([x[k] for x in lst])
        else
            v = [x[k] for x in lst]
        end
        ret[k] = v
    end
    return ret
end