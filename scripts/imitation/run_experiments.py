
import argparse
import os 
import shlex
import sys

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-n', '--dry-run', action='store_true')

def new_session(session, window):
    cmds = []
    cmds += ['tmux kill-session -t {}'.format(session)]
    cmds += ['tmux new-session -s {} -n {} -d bash'.format(session, window)]
    cmds += ['sleep 1']
    return cmds

def new_window(session, window):
    return 'tmux new-window -t {} -n {} bash'.format(session, window)

def new_cmd(session, window, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join(shlex.quote(str(v)) for v in cmd)
    return 'tmux send-keys -t {}:{} {} Enter'.format(session, window, shlex.quote(cmd))

def new_tensorboard_cmds(session, window, port, expdir):
    cmds = []
    cmds += [new_window(session, window)]
    cmds += ['sleep 1']
    cmds += [new_cmd(session, window, ['source', 'activate', 'rllab3'])]
    logdir = os.path.join(expdir, 'imitate', 'summaries')
    cmds += [new_cmd(session, window, ['tensorboard', '--logdir', logdir, '--port', port])]
    return cmds

def new_activate_cmd(session, window):
    return new_cmd(session, window, ['source', 'activate', 'rllab3'])

def build_gail_cmds(basedir, n_itr=2000, expname='gail', port='55555'):
    cmds = []
    session = expname
    expdir = os.path.join(basedir, expname)
    cmds += new_session(session, 'train')
    cmds += [new_activate_cmd(session, 'train')]
    cmds += [new_cmd(session, 'train', [
        'python', 'imitate.py', 
        '--exp_name', expname,
        '--use_infogail', 'False',
        '--n_itr', n_itr
    ])]
    cmds += new_tensorboard_cmds(session, 'tb', port, expdir)
    return cmds

def build_infogail_cmds(basedir, n_itr=1000, expname='infogail', port='55554'):
    cmds = []
    session = expname
    expdir = os.path.join(basedir, expname)
    cmds += new_session(session, 'train')
    cmds += [new_activate_cmd(session, 'train')]
    cmds += [new_cmd(session, 'train', [
        'python', 'imitate.py', 
        '--exp_name', expname,
        '--use_infogail', 'True',
        '--n_itr', n_itr
    ])]
    cmds += new_tensorboard_cmds(session, 'tb_infogail', port, expdir)
    params_filepath = os.path.join(expdir, 'imitate', 'log', 'itr_{}.npz'.format(n_itr))
    return cmds, expdir, params_filepath

def build_hgail_cmds(
        basedir, 
        params_filepath, 
        n_itr=1000,
        session='infogail', 
        expname='hgail', 
        port='55553'):
    '''
    run hgail after infogail, using the session and params filepath from infogail
    '''
    cmds = []
    expdir = os.path.join(basedir, expname)
    cmds += [new_cmd(session, 'train', [
        'python', 'imitate_hgail.py', 
        '--exp_name', expname,
        '--params_filepath', params_filepath,
        '--n_itr', n_itr
    ])]
    cmds += new_tensorboard_cmds(session, 'tb_hgail', port, expdir)
    return cmds

def build_cmds():
    basedir = '../../data/experiments/'
    gail_cmds = build_gail_cmds(basedir)
    infogail_cmds, infogail_dir, params_filepath = build_infogail_cmds(basedir)
    hgail_cmds = build_hgail_cmds(basedir, params_filepath)
    cmds = gail_cmds + infogail_cmds + hgail_cmds
    return cmds

if __name__ == '__main__':
    args = parser.parse_args()
    cmds = build_cmds()
    print('\n'.join(cmds))
    if args.dry_run:
        print('\nabove commands would be run if dry-run not set')
    else:
        os.environ["TMUX"] = ""
        os.system('\n'.join(cmds))

