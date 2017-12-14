using AutomotiveDrivingModels
using AutoRisk
using HDF5
using NGSIM

# extraction settings and constants
timestep_delta = 1 # timesteps between feature extractions
record_length = 20 # number of frames for record to track in the past
offset = 100 # from ends of the trajectories
maxframes = 5000 # nothing for no max

output_filename = "ngsim.h5"
output_filepath = joinpath("../data/trajectories/", output_filename)

models = Dict{Int, DriverModel}() # dummy, no behavior available
println("output filepath: $(output_filepath)")

# feature extractor (note the lack of behavioral features)
# use these feature extractors for lidar imputation:
subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        CarLidarFeatureExtractor(20, carlidar_max_range = 50.)
    ]

ext = MultiFeatureExtractor(subexts)
n_features = length(ext)
features = Dict{Int, Dict{Int, Array{Float64}}}()

tic()
n_traj = 1
# extract 
for traj_idx in 1:n_traj

    # setup
    trajdata = load_trajdata(traj_idx)
    roadway = get_corresponding_roadway(traj_idx)
    max_n_objects = maximum(n_objects_in_frame(trajdata, i) for i in 1 : nframes(trajdata))
    scene = Scene(max_n_objects)
    rec = SceneRecord(record_length, 0.1, max_n_objects)
    features[traj_idx] = Dict{Int, Array{Float64}}()
    ctr = 0

    for frame in offset : (nframes(trajdata) - offset)
        ctr += 1
        if maxframes != nothing && ctr >= maxframes
            break
        end

        print("\rtraj: $(traj_idx) / $(n_traj)\tframe $(frame) / $(nframes(trajdata) - offset)")
            
        # update the rec
        AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))

        # every timestep_delta step, extract features
        if frame % timestep_delta == 0

            for (vidx, veh) in enumerate(scene)
                # extract features
                veh_features = pull_features!(ext, rec, roadway, vidx, models)
                
                # add entry to features if vehicle not yet encountered
                if !in(veh.id, keys(features[traj_idx]))
                    features[traj_idx][veh.id] = zeros(n_features, 0)
                end

                # stack onto existing features
                features[traj_idx][veh.id] = cat(2, features[traj_idx][veh.id], 
                    reshape(veh_features, (n_features, 1)))
            end
        end
    end
end
toc()

# compute max length across samples
maxlen = 0
for (traj_idx, feature_dict) in features
    for (veh_id, veh_features) in feature_dict
        maxlen = max(maxlen, size(veh_features, 2))
    end
end
println("max length across samples: $(maxlen)")

# write trajectory features
h5file = h5open(output_filepath, "w")
for (traj_idx, feature_dict) in features

    feature_array = zeros(n_features, maxlen, length(feature_dict))
    for (idx, (veh_id, veh_features)) in enumerate(feature_dict)
        feature_array[:, 1:size(veh_features, 2), idx] = reshape(veh_features, (n_features, size(veh_features, 2), 1))
    end
    h5file["$(traj_idx)"] = feature_array

end

# write feature names
attrs(h5file)["feature_names"] = feature_names(ext)
close(h5file)