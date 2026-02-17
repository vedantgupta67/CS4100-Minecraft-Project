# CS4100-Minecraft-Project


# Installation requirements for MineRL


need to clone minerl into minerl folder
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

Should be able to run code after that!