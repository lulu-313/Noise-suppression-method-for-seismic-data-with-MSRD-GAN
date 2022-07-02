import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_h5,save_h5
from torchvision.transforms import ToTensor,Resize
from PIL import Image
from skimage.transform import resize  


class Dataset_generator(Dataset):
    def __init__(self,data_path,normal = True,argumentation = True,train = True,sigma=25):
        """

        Args:
            data_path: Seismic data paths without noise
            normal: Normalize to 0,1
            argumentation: Whether to use data enhancement
            train: training or not
        """
        super(Dataset,self).__init__()
        self.files_path = [os.path.join(data_path,x) for x in os.listdir(data_path)]
        self.argumentation = argumentation
        self.normal = normal
        self.train = train
        self.sigma = sigma

    def __getitem__(self,index):
        raw_data = read_h5(self.files_path[index])
        noise = torch.randn(raw_data.shape).mul_(self.sigma/255.0).numpy()    
        noise_data = raw_data + noise
        # low_data = read_h5(self.low_files_path[index])
        restore_size = raw_data.shape

        if self.argumentation:
            noise_data,raw_data = self._argumentation(noise_data,raw_data)

        if self.normal:
            noise_data = self._normal(noise_data)
            raw_data = self._normal(raw_data)

        to_tensor = ToTensor()
        return to_tensor(noise_data).type(torch.FloatTensor),to_tensor(raw_data).type(torch.FloatTensor)
        
    def __len__(self):
        return len(self.files_path)

    def _normal(self,data):
        data_max = np.max(data)
        data_min = np.min(data)
        normal = (data - data_min) / (data_max - data_min)
        return normal

    # flip or rot 180 angle
    def _argumentation(self,noise_data,raw_data):
        if random.random() > 0.5:
            raw_data = np.flip(raw_data, axis=0)
            noise_data = np.flip(noise_data, axis=0)
        if random.random() > 0.5:
            raw_data = np.flip(raw_data, axis=1)
            noise_data = np.flip(noise_data, axis=1)
        if random.random() > 0.5:
            raw_data = np.rot90(raw_data, k=2)
            noise_data = np.rot90(noise_data, k=2)
        return noise_data,raw_data

