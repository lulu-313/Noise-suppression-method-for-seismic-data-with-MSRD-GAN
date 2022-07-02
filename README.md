# Noise-suppression-method-for-seismic-data-with-MSRD-GAN

An application of MSRD-GAN in noise suppression of seismic data. This is a repository for the paper "Noise suppression method for seismic data based on multi-scale residual dense network and adversarial learning".


## Example
**Actual data processing：**
![image](https://github.com/lulu-313/Noise-suppression-method-for-seismic-data-with-MSRD-GAN/blob/master/apply.png)

**waveform：**
![image](https://github.com/lulu-313/Noise-suppression-method-for-seismic-data-with-MSRD-GAN/blob/master/waveform.png)



## Project Organization

![image](https://github.com/lulu-313/Noise-suppression-method-for-seismic-data-with-MSRD-GAN/blob/master/file-structure.png)


## Code

All training and test code are in the directory **code**.

## Dataset

The the filed processing data by our method are in **data/filed_apply** folder.
The synthetic seismic data used for training can be obtained by visting the "https://drive.google.com/drive/folders/1V3LHqGFH6NNS-GsusE_Ncz6WUAtgtNH3?usp=sharing".

## Dependencies

* python 3.6.13
* pytorch 1.9.0
* torchvision 0.10.0
* tqdm 4.64.0
* scipy 1.4.1
* numpy 1.19.5
* h5py 2.10.1
* pandas 1.1.5
* matplotlib 3.1.2

* cuda 10.1


## Usage instructions
Download this project and build the dependency.
Then use application_filed.py --test_data_path="your data path" --save_path="your save path of resed by GAN"

## Citation

If you find this work useful in your research, please consider citing this project.