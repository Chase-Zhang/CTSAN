# CTSAN

### Cascaded Temporal and Spatial Attention Network for Solar AO Image Restoration


An overview of proposed CTSAN architecture. Panel (a) shows the network detail of the TSAN unit. Panel (b) is the input and output of a
single TSAN unit. Panel (c) shows the forward process of CTSAN. Only one set of TSAN parameter is trained during backward propagation and then used four times in forward propagation to construct the cascaded two-stage architecture. 


![CTSAN](./img_display/CTSAN.png)


![results](./img_display/result_5th.png)

CTSAN has a stable performance on the lowest granulation contrast frames of TiO AO closed-loop images captured by NVST telescope with GLAO correction system, indicating our cascaded network may has the potential to maintain a stable performance in actual astronomical observation conditions.


### Environments and Dependencies

We run CTSAN model on Linux sytem with configuration of Ubuntu 20.04, CUDA 11.7, and GPU RTX 3090.  
```
Python 3 (Anaconda is preferred)  

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
### Download

1. Please download the PWC-Net Pretrained parameter frome [here]( ) if your want to train/test CTSAN.  

   After download the file named network-default.pytorch, please put it into this folder: "./pretrain_models".  
		


2. Please download it from [here]() if you want to use the CTSAN parameters trained on our NVST real AO dataset.  

   After download, please put them into this folder: "./Trained_Model".
   
   
   
3. Please download it from [here]() if you want to use our NVST real AO testing dataset.  

   After download, please put them into this folder: "./dataset".



### Dataset Organization

Please organize your prepared training/validation/testing sets ( .img or .jpg image format ) following this catalog structure:
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


### Training

```
cd ./code

python main.py --save  path_to_save_result  --dir_data train_data_path   --dir_data_test validation_data_path  --epochs total_epoch_number  --batch_size 8
``` 

The result will be saved in "./experiment".  

If you want to get some intermediate results, please set the optional item save_images in the path of "./code/option/init.py" as True, which will greatly prolong the training time.


### Inference

```
cd ./code

python inference.py  --data_path  test_data_path  --model_path saved_model_parameter_path  
```

The result will be saved in "./infer_results".

If you want to save the restored results as image formats, please set the optional item save_images in the path of "./code/inference.py" as True, which will greatly prolong the inference time.


### Citation

@article{zhang2023cascaded,  
  title={Cascaded Temporal and Spatial Attention Network for Solar Adaptive Optics Image Restoration},  
  author={Chi Zhang*, Shuai Wang*, Libo Zhong, Qingqing Chen, Changhui Rao},  
  journal={Astronomy \& Astrophysics},  
  volume={ },  
  pages={ },  
  year={2023},  
  publisher={EDP Sciences}  
}


### Contact Us

Please send email to zhangchi.ch@gmail.com or wangshuai0601@uestc.edu.cn.
