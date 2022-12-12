import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AutoFusion(nn.Module):

    def __init__(self, input_features, latent_features, num_classes=5):

        super().__init__()

        inp1 = input_features // 2

        # fuse in
        self.linear1 = nn.Linear(input_features, inp1)
        self.linear2 = nn.Linear(inp1, latent_features)

        # classification layer
        self.classification = nn.Linear(latent_features, num_classes)

        # fues_out
        self.linear3 = nn.Linear(latent_features, inp1)
        self.linear4 = nn.Linear(inp1, input_features)

        # weight tying
        self.linear3.weight = nn.Parameter(self.linear2.weight.T)
        self.linear4.weight = nn.Parameter(self.linear1.weight.T)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, concat_embd):

        # fuse in
        fused = self.tanh(self.linear1(concat_embd.to(device)))
        fused = self.relu(self.linear2(fused.to(device)))

        # classificatin probablities
        class_probs = self.classification(fused)

        # fuse_out
        reconst = self.tanh(self.linear3(fused.to(device)))
        reconst = self.linear4(reconst.to(device))

        return fused, reconst, class_probs

def train(camera_dataloader, camera_model, sonar_dataloader, sonar_model, autofusion_model, criterion_mse, criterion_ce, optimizer):
  
    camera_model.eval()
    sonar_model.eval()
    autofusion_model.train()

    total_loss = 0.0
    num_correct=0
    for i, data in tqdm(enumerate(zip(camera_dataloader, sonar_dataloader))):

        # Move images to device
        camera_img = data[0][0].to(device)
        camera_label = data[0][1].to(device)
        sonar_img = data[1][0].to(device)
        sonar_label = data[1][1].to(device)

        # Get model outputs
        camera_embd = camera_model(camera_img)
        sonar_embd = sonar_model(sonar_img)

        #print(camera_embd)
        #print(sonar_embd)
        
        concat_embd = torch.concat((camera_embd, sonar_embd), dim=1)
        
        _, reconst_embd, class_probs = autofusion_model(concat_embd)

        class_loss = criterion_ce(class_probs, camera_label)
        reconstruction_loss_mse = criterion_mse(concat_embd, reconst_embd)
 
        num_correct += int((torch.argmax(class_probs, axis=1) == camera_label).sum())
        total_loss += float(reconstruction_loss_mse.item()+class_loss.item())

        optimizer.zero_grad()
        (reconstruction_loss_mse+class_loss).backward()
        optimizer.step()

    
    acc = 100 * num_correct / (8* len(camera_dataloader))
    total_loss = float(total_loss / len(sonar_dataloader))

    return total_loss,reconst_embd, acc

def validate(camera_dataloader, camera_model, sonar_dataloader, sonar_model, autofusion_model, criterion_mse, criterion_ce, optimizer):
  
    autofusion_model.eval()
    camera_model.eval()
    sonar_model.eval()
    #batch_bar = tqdm(total=len(camera_dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0.0
    total_loss = 0.0


    for i, data in tqdm(enumerate(zip(camera_dataloader, sonar_dataloader))):

        # Move images to device
        camera_img = data[0][0].to(device)
        camera_label = data[0][1].to(device)
        sonar_img = data[1][0].to(device)
        sonar_label = data[1][1].to(device)

        # Get model outputs
        with torch.inference_mode():
          camera_embd = camera_model(camera_img)
          sonar_embd = sonar_model(sonar_img)

        #print(camera_embd)
        #print(sonar_embd)
        
        concat_embd = torch.concat((camera_embd, sonar_embd), dim=1)
        
        with torch.inference_mode():
          _, reconst_embd, class_probs = autofusion_model(concat_embd)

        class_loss = criterion_ce(class_probs, camera_label)
        reconstruction_loss_mse = criterion_mse(concat_embd, reconst_embd)
 
        num_correct += int((torch.argmax(class_probs, axis=1) == camera_label).sum())
        total_loss += float(reconstruction_loss_mse.item()+class_loss.item())

        #optimizer.zero_grad()
        #(reconstruction_loss_mse+class_loss).backward()
        #optimizer.step()
        
    #batch_bar.close()
    acc = 100 * num_correct / (8* len(camera_dataloader))
    total_loss = float(total_loss / len(sonar_dataloader))
    return acc, total_loss



def train_autofusion_model(train_loader_camera,train_loader_sonar,val_loader_camera,val_loader_sonar,camera_model,sonar_model):
    #camera_model = Network().to(device)
    #camera_model.load_state_dict(torch.load("/content/drive/MyDrive/IDL Project/camera_model_final.pt"))
    camera_model_fuse = torch.nn.Sequential(*(list(camera_model.children())[:-1]))
    #sonar_model =Network().to(device)
    #sonar_model.load_state_dict(torch.load("/content/drive/MyDrive/IDL Project/sonar_model_final.pt"))
    sonar_model_fuse = torch.nn.Sequential(*(list(sonar_model.children())[:-1]))
    autofusion_model = AutoFusion(1024, 256).to(device)

    optimizer = torch.optim.AdamW(autofusion_model.parameters(), lr=1e-4, weight_decay=1e-2)

    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    for epoch in range(30):

        train_loss,reconst_emb,train_acc = train(train_loader_camera, camera_model_fuse, train_loader_sonar, sonar_model_fuse, autofusion_model, criterion_mse, criterion_ce, optimizer)
        val_acc,val_loss=validate(val_loader_camera,camera_model_fuse,val_loader_sonar,sonar_model_fuse,autofusion_model,criterion_mse, criterion_ce, optimizer)
        
        print("Epoch: ",epoch)
        print("Training Loss is ",train_loss)
        print("Training Acc is ",train_acc)
        print("Validation Loss is ",val_loss)
        print("Validation Acc is ",val_acc)

        print("-"*10)
    
    return autofusion_model