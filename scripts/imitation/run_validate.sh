#!/bin/bash

python validate.py --n_proc 12 --exp_dir ../../data/experiments/0_gail/ --params_filename itr_2000.npz --n_runs_per_ego_id 2 
python validate.py --n_proc 12 --exp_dir ../../data/experiments/0_infogail/ --params_filename itr_1000.npz --n_runs_per_ego_id 2
python validate.py --n_proc 12 --exp_dir ../../data/experiments/0_recurrent_gail/ --params_filename itr_2000.npz --n_runs_per_ego_id 2
python validate.py --n_proc 12 --exp_dir ../../data/experiments/0_hgail/ --params_filename itr_999.npz --n_runs_per_ego_id 2 --use_hgail True