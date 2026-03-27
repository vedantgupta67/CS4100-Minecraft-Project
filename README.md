# CS4100-Minecraft-Project


# Installation requirements for MineRL


need to clone minerl into minerl folder:

git clone https://github.com/minerllabs/minerl.git minerl

python 3.10:

run:

brew install python@3.10

python3.10 -m venv venv

source venv/bin/activate

confirm with: python --version


java 8

run:

brew install --cask temurin@8

export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home

confirm with: java --version


Then from home directory run:
pip install setuptools==65.5.0
pip install gym==0.23.1
pip install -e minerl/

confirm installation with: pip list


## Windows Specific
Install WSL Ubuntu version >22 and Temurin Java 8 onto it

Should be able to run code after that!


## Installing MineRL
MineRL can take up to 30 minutes when installing, stay patient.

## VPT Model
To run the wood_crafting_agent.py file:

git clone https://github.com/openai/Video-Pre-Training.git

From https://github.com/openai/Video-Pre-Training

Download 1x Model and 1x Width Weights from their README.md

Place those files into /scripts

By default wood_crafting_agent.py uses those files, if you want to specify a different path use the flags --vpt-model {path} and --vpt-weights {path}

You can use different the 2x or 3x model and width weights, however, those are much larger and slower, but can produce better results than the 1x.
## Checkpoints

By default, wood_crafting_agent.py saves a .pth file to the /checkpoints folder every 50,000 steps

To change the save directory for checkpoints: --save-dir {directory name} (default: checkpoints)

You can change this using the --save-every {# of steps} flag

To run a checkpoint, use the --resume {path} flag

To evaluate a checkpoint, use the --eval {path} flag

## Other Useful Arguments
To change the total number of timesteps: --timesteps {# of timesteps} (default: 500,000)

To change the rollout of stats: --rollout-steps {# of rollout-steps} (default: 2048)

To number of episodes: --episodes {# of episodes} (default: 5)

To print the rewards gained or lost at every step: --print-rewards (default: false)


