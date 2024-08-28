## Installing nidaqmx package and NI-DAQmx driver on Ubuntu 20.04
```bash
sudo apt install ni-daqmx
pip install nidaqmx
sudo apt-get -y install ni-hwcfg-utility
sudo dkms autoinstall
```

## Pre-req for SAMV2
- Requires having checkpoints in `./checkpoints` folder in the workspace directory.