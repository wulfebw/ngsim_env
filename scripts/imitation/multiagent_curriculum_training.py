
import argparse
import os
import subprocess

def build_commands(
        exp_name,
        n_itr_each,
        n_envs_start,
        n_envs_end,
        n_envs_step,
        exp_dir='../../data/experiments'):
    # template command to be completed for each individual run
    # (this syntax creates a single string, used for interpretable formatting)
    template = ('python imitate.py '
        '--env_multiagent True '
        '--use_infogail False '
        '--exp_name {} '
        '--n_itr {} '
        '--policy_recurrent True '
        '--n_envs {} '
        '--params_filepath {} '
        '--validator_render False' # avoid bug where render takes a very long time
    )
    cmds = []
    # explicit empty string for initial run
    params_filepath = "''" 
    for n_envs in range(n_envs_start, n_envs_end + n_envs_step, n_envs_step):
        # each command differs only in the experiment name and the params_filepath
        cmd = template.format(
            exp_name.format(n_envs), 
            n_itr_each, 
            n_envs,
            params_filepath
        )
        cmds.append(cmd)

        # update params filepath
        params_filepath = os.path.join(
            exp_dir,
            exp_name.format(n_envs),
            'imitate/log/itr_{}.npz'.format(n_itr_each)
        )
    return cmds
    
def run_commands(cmds, dry_run):
    for cmd in cmds:
        print(cmd)
        if not dry_run:
            subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='multiagent_curriculum_{}')
    parser.add_argument('--n_itr_each', type=int, default=200)
    parser.add_argument('--n_envs_start', type=int, default=10)
    parser.add_argument('--n_envs_end', type=int, default=50)
    parser.add_argument('--n_envs_step', type=int, default=10)
    parser.add_argument('--dry_run', action='store_true', default=False)
    args = parser.parse_args()

    # build commands
    cmds = build_commands(
        args.exp_name, 
        args.n_itr_each,
        args.n_envs_start,
        args.n_envs_end,
        args.n_envs_step,
    )

    # run commands
    run_commands(cmds, args.dry_run)
