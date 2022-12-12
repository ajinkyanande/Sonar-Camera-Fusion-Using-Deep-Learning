import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torchvision
from skimage.util import random_noise


#Custom Transform for adding noise
class Add_Noise(object):

    def __init__(self,mode):
      assert isinstance(mode,str)
      self.mode=mode

    def __call__(self, image):
      image = torch.tensor(random_noise(image, mode=self.mode, mean=0, var=0.1, clip=True),dtype=torch.float32)   
      #print(image)
      return image


def load_data(path_to_camera_data,path_to_sonar_data):
    transforms_camera=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),Add_Noise("gaussian")])
    transforms_sonar=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    camera_dataset = torchvision.datasets.ImageFolder(root=path_to_camera_data,transform=transforms_camera)
    sonar_dataset = torchvision.datasets.ImageFolder(root=path_to_sonar_data, transform=transforms_sonar)

    train_size = int(0.8 * len(camera_dataset))
    val_size = len(camera_dataset) - train_size

    camera_train, camera_test = torch.utils.data.random_split(camera_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    sonar_train, sonar_test = torch.utils.data.random_split(sonar_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

    train_loader_sonar = torch.utils.data.DataLoader(sonar_train, batch_size = 8, 
                                            shuffle = False,num_workers = 2, pin_memory = True)
    val_loader_sonar = torch.utils.data.DataLoader(sonar_test, batch_size = 8, 
                                            shuffle = False, num_workers = 2)
    train_loader_camera = torch.utils.data.DataLoader(camera_train, batch_size = 8, 
                                            shuffle = False,num_workers = 2, pin_memory = True)
    val_loader_camera = torch.utils.data.DataLoader(camera_test, batch_size = 8, 
                                            shuffle = False, num_workers = 2)

    return train_loader_sonar,train_loader_camera,val_loader_sonar,val_loader_camera

