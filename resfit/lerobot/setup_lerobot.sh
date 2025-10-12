# Git clone lerobot into home directory
git clone https://github.com/huggingface/lerobot.git ~/lerobot
git -C ~/lerobot checkout 69901b9b6a2300914ca3de0ea14b6fa6e0203bd4

# Install lerobot
python -m pip install -e ~/lerobot --no-deps

# Install a couple of dependencies
python -m pip install -r resfit/lerobot/lerobot_requirements.txt
python -m pip install --upgrade torch torchvision torchcodec
python -m pip install datasets==3.6.0