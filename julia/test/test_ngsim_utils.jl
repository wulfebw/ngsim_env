using Base.Test
using AutoEnvs

function test_index_ngsim_trajectory()
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0750am-0805am.txt")
    index = index_ngsim_trajectory(filepath, verbose=1)
    @test length(index) > 2000
    for (k,v) in index
        @test in("ts", keys(v))
        @test in("te", keys(v))
    end
end

function test_sample_trajdata_vehicle()
    trajinfos = [Dict(1=>Dict("ts"=>1, "te"=>2))]
    traj_idx, vehid, ts, te = sample_trajdata_vehicle(trajinfos)
    @test traj_idx == 1
    @test vehid == 1
    @test ts >= 1
    @test te == 2
end

function test_build_feature_extractor()
    ext = build_feature_extractor()
end

function test_load_ngsim_trajdatas()
    filenames = [
        "trajdata_i80_trajectories-0400-0415.txt",
        "trajdata_i80_trajectories-0500-0515.txt",
        "trajdata_i80_trajectories-0515-0530.txt",
        "trajdata_i101_trajectories-0805am-0820am.txt",
        "trajdata_i101_trajectories-0820am-0835am.txt",
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]
    filepaths = [Pkg.dir("NGSIM", "data", fn) for fn in filenames]
    load_ngsim_trajdatas(filepaths, minlength=250)

end

@time test_index_ngsim_trajectory()
@time test_sample_trajdata_vehicle()
@time test_build_feature_extractor()
@time test_load_ngsim_trajdatas()