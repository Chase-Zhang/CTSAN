# Cascaded Temporal and Spatial Attention Network for Solar Adaptive Optics Image Restoration (CTSAN)


## Environments and Dependencies
```
Linux (Ubuntu 20.04, CUDA 11.7, RTX 3090)  

Python 3 (Anaconda is Preferred)  

Pytorch:   
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
numpy: 
	conda install numpy
matplotlib: 
	conda install matplotlib
opencv: 
	conda install oepncv
imageio:
	conda install imageio
skimage: 
	conda install scikit-image
tqdm: 
	conda install tqdm
cupy: 
	pip install cupy-cuda117 or conda install -c anaconda cupy
```
## Download
1. Please download the PWC-Net Pretrained model frome [here](https://github.com/sniklaus/pytorch-pwc) if your want to train/test CTSAN.   
  
		After download the file named network-default.pytorch, please put it into this folder: "./pretrain_models".  
		
		
2. If you want to use the CTSAN parameters trained on our NVST real AO dataset, please download it from [here]().  

		After download, please put them into this folder: "./Trained_Model".
	

## Dataset Organization
Please organize your prepared training/validation/testing sets following this catalog structure:
```
|--dataset
   |--blur
      |--burst 1
	  |--frame 1
	  |--frame 2
	  |--   .
	  |--   .
	  |--frame n
      |--burst 2
      |--    .
      |--    .
      |--burst n
   |--gt
      |--burst 1
          |--frame 1
          |--frame 2
          |--   .
          |--   .
          |--frame n
      |--burst 2
      |--   .
      |--   .
      |--burst n
		

```


## Training

## Inference

## Cition
