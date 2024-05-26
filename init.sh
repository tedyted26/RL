sudo apt-get update
sudo apt-get --allow-releaseinfo-change update -y
sudo apt-get install ffmpeg libsm6 libxext6  -y

pip install gym[atari]
pip install autorom
pip install gym[accept-rom-license]

pip install matplotlib
pip install torch
pip install torchbnn
pip install opencv-python
pip install numpy==1.23.1

#Use 1.85.2 coder
