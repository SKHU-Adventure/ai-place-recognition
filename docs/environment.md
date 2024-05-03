# NVIDIA Driver 535 and other packages
```bash
sudo apt update
sudo apt upgrade
suto apt install git
sudo apt install python3-venv
sudo apt install nvidia-driver-550
sudo reboot
```

# CUDA 12.1
You should check CUDA Toolkit 12.1 only!!!!!!!!!!
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```
Add following lines to ~/.bashrc
```
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
```
Then apply the changes
```bash
source ~/.bashrc
```

# CUDNN 9.1.1
```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz
```
TBD

# NCCL 2.12.12
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo sudo apt install libnccl2=2.18.3-1+cuda12.1 libnccl-dev=2.18.3-1+cuda12.1
```

# Assign github
Generate key
```bash
ssh-keygen
```
Print the key and Apply to github
```bash
cat ~/.ssh/id_rsa.pub
```
