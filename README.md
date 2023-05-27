# StableCFR

## Env:

```shell
conda create -n tf-torch-gpu python=3.7 -y
conda activate tf-torch-gpu
conda install pysocks -y
conda install tensorflow-gpu==1.15.0
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install scipy pandas matplotlib
pip install --upgrade protobuf==3.20.1
pip install --upgrade numpy==1.18
pip install seaborn scikit-learn
```

Hardware used: Ubuntu 16.04.5 LTS operating system with 2 * Intel Xeon E5-2678 v3 CPU, 384GB of RAM, and 4 * GeForce GTX 1080Ti GPU with 44GB of VRAM.

Software used: Python 3.7.15 with TensorFlow 1.15.0, Pytorch 1.7.1, NumPy 1.18.0, and MatplotLib 3.5.3.

## Run:

```shell
conda activate tf-torch-gpu
python main.py
```