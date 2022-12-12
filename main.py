import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

from data_loading import load_data
from baselines import train_baseline_models
from baseline_fusion import train_baseline_fusion
from autofusion_model import train_autofusion_model
from gan_fusion_model import train_ganfusion_model

if __name__=="__main__":
    
    #load the images into dataloaders
    train_loader_sonar,train_loader_camera,val_loader_sonar,val_loader_camera=load_data("sonar_camera_data/camera","sonar_camera_data/sonar")
    #train unimodal baseline models
    #camera_model,sonar_model=train_baseline_models(train_loader_camera,train_loader_sonar,val_loader_camera,val_loader_sonar)
    #train baseline fusion model
    concatenated_embeddings_model=train_baseline_fusion(train_loader_camera,train_loader_sonar,val_loader_camera,val_loader_sonar)
    #train autofusion model
    #autofusion_model=train_autofusion_model(train_loader_camera,train_loader_sonar,val_loader_camera,val_loader_sonar,camera_model,sonar_model)
    #train ganfusion model
    #ganfusion_model=train_ganfusion_model(train_loader_camera,train_loader_sonar,val_loader_camera,val_loader_sonar,camera_model,sonar_model)


