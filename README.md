
# NGSIM Env
- This is a rllab environment for learning human driver models with imitation learning
- This repository does not contain a gail / infogail / hgail implementation
- It also does not contain the human driver data you need for the environment to work. See [NGSIM.jl](https://github.com/sisl/NGSIM.jl) for that.

## Demo
- Hierarchical GAIL (implementation not included) in a single-agent environment
![alt tag](https://raw.githubusercontent.com/wulfebw/ngsim_env/master/media/ngsim_env_hgail.gif)

# Install

# Documentation

## How's this work?
- See README files individual directories for details, but a high-level description is:
- The python code uses [pyjulia](https://github.com/JuliaPy/pyjulia) to instantiate a Julia interpreter, see the `python` directory for details
- The driving environment is then built in Julia, see the `julia` directory for details
- Each time the environment is stepped forward, execution passes from python to julia, updating the environment
