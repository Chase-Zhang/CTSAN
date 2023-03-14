# Cascaded Temporal and Spatial Attention Network for Solar Adaptive Optics Image Restoration (CTSAN)


## Environments and Dependencies
```
Linux (Ubuntu 20.04, CUDA 11.7, RTX 3090)
Python 3 (Anaconda is Preferred)
Pytorch: conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
numpy: conda install numpy
matplotlib: conda install matplotlib
opencv: conda install oepncv
imageio; conda install imageio
skimage: conda install scikit-image
tqdm: conda install tqdm
cupy: pip install cupy-cuda117 or conda install -c anaconda cupy
```
## Download


## Dataset Organization
Please organize your prepared training/validation/testing sets following this catalog structure:  \<br>

dataset  \<br>
	blur\<br>
			burst 1\<br>
					frame 1 \<br>
					frame 2\<br>
					.\<br>
					.\<br>
					frame n\<br>
			burst 2\<br>
	gt\<br>
			burst 1\<br>
					.\<br>
					.\<br>
			burst 2\<br>
					.\<br>
					.\<br>



## Training

## Inference

## Cition
