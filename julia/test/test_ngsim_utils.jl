# using Base.Test
# using AutoEnvs

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

@time test_index_ngsim_trajectory()
@time test_sample_trajdata_vehicle()
@time test_build_feature_extractor()