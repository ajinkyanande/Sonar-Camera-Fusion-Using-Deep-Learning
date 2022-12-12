import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = {
    'latent_dim': 512,
    'hidden_dim': 256,
    'lr': 1e-3
}

class GanDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim: dimension of the input latent codes
        hidden_dim: classifier's hidden dimension
        """
        super(GanDiscriminator, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  nn.BatchNorm1d(hidden_dim),  nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),  nn.BatchNorm1d(hidden_dim),  nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)  
        )
        self._loss =  nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.Adam(self.parameters(), config['lr'])


    def optimize(self, clip):
        nn.utils.clip_grad_norm_(self.parameters(), clip)
        self._optimizer.step()


    def forward(self, input_latent, labels=None):
        output = self._classifier(input_latent)
        output_dict = {'output': output}
        if labels is not None:
            output_dict['loss'] = self._loss(output, labels)
            output_dict['accuracy'] = (torch.sigmoid(output).round() == labels).float().mean()
        return output_dict


class GanGenerator( nn.Module):
    def __init__(self, latent_dim, temperature=0.01):
        super(GanGenerator, self).__init__()
        self._latent_mapper =  nn.Sequential(
             nn.Linear(latent_dim, 2*latent_dim),  nn.BatchNorm1d(2*latent_dim),  nn.LeakyReLU(0.2),
             nn.Linear(2*latent_dim, 2*latent_dim),  nn.BatchNorm1d(2*latent_dim),  nn.LeakyReLU(0.2),
             nn.Linear(2*latent_dim, latent_dim)
        )

        self._temperature = temperature
        self._ce_loss =  nn.BCEWithLogitsLoss()
        self._optimizer =  torch.optim.Adam(self.parameters(), config['lr'])

    def add_noise(self, latent):
        """
        Adds standard normal noise to a latent vector
        """
        return self._temperature*torch.randn(latent.shape).to(device) + latent

    def optimize(self, clip):
        nn.utils.clip_grad_norm_(self.parameters(), clip)
        self._optimizer.step()

    def forward(self, target_latent, discriminator):
        # add noise to input data
        target_latent = self.add_noise(target_latent)
        z_g = self._latent_mapper(target_latent)
        output_dict = {}
        # continue discriminator part
        
        if discriminator is not None:
            predicted = discriminator(z_g)['output']
            # the generator wants the disc to think these as true
            desired = torch.ones_like(predicted[:,0].reshape(-1,1))
            # print(predicted.shape, desired.shape)
            output_dict['loss'] = self._ce_loss(predicted, desired)
            output_dict['accuracy'] = (torch.sigmoid(predicted).round() == desired).float().mean()

        output_dict.update({
            'z_g': z_g
            })
        return output_dict

class GanFusionSingle(nn.Module):
    """docstring for GanFusionSingle: fusion module for one target modality"""
    def __init__(self,
                 config,
                 generator,
                 discriminator):
        """
        target_latent: target modality's latent code
        compl_latent: complementary modalities' latent codes (concatenated)
        """
        super(GanFusionSingle, self).__init__()
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.gen_metrics = None
        self.disc_metrics = None

    def forward(self, z_dict, mode, stage=None):
        # refer to the paper for conventions
        if mode != 'train':
            stage = 'gen'

        if stage == 'disc_real':
            z_tr = z_dict['compl']
            labels = torch.ones_like(z_tr[:,0].reshape(-1,1))
            output = self.discriminator(z_tr, labels)
            # add autofusion's loss as well

            output['z_g'] = self.generator(z_dict['target'], self.discriminator)['z_g']
            self.disc_metrics = {
                'drl': output['loss'],
                'dracc': output['accuracy']
            }
        elif stage == 'disc_fake':
            # get the generated latent z_g
            z_g = self.generator(z_dict['target'], self.discriminator)['z_g']
            labels = torch.zeros_like(z_g[:,0].reshape(-1,1))
            output = self.discriminator(z_g, labels)
            output['z_g'] = z_g
            self.disc_metrics = {
                'dfl': output['loss'],
                'dfacc': output['accuracy']
            }
        elif stage == 'gen':
            output = self.generator(z_dict['target'], self.discriminator)
            self.gen_metrics = {
                'gl': output['loss'],
                'gacc': output['accuracy']
            }
        else:
            raise ValueError(f'invalid stage: {stage}')
        return output


class GanFusion(nn.Module):
    """docstring for GanFusion: contains GAN-Fusion module for all the
    modalities"""
    def __init__(self, config):
        super(GanFusion, self).__init__()
        self.sonar_gan = GanFusionSingle(config,
                                        GanGenerator(config['latent_dim']),
                                        GanDiscriminator(config['latent_dim'],
                                                        config['hidden_dim']))
        self.camera_gan = GanFusionSingle(config,
                                          GanGenerator(config['latent_dim']),
                                          GanDiscriminator(config['latent_dim'],
                                                        config['hidden_dim']))

        self.ff_input_features = config['latent_dim']*2
        self.feed_forward = nn.Sequential(
            nn.Linear(self.ff_input_features, self.ff_input_features//2),
            nn.Tanh(),
            nn.Linear(self.ff_input_features//2, config['latent_dim']),
            nn.ReLU()
            )

    def z_fuse(self, fusion_dict):
        z_fuse_t = fusion_dict['sonar_dict']['z_g']
        z_fuse_s = fusion_dict['camera_dict']['z_g']
        return self.feed_forward(torch.cat((z_fuse_t,
                                            z_fuse_s), dim=-1))

    def get_loss(self, fusion_dict):
        return fusion_dict['sonar_dict']['loss'] + \
               fusion_dict['camera_dict']['loss']

    def forward(self, latent_dict, mode, stage=None):
        z_t = latent_dict['z_sonar']
        z_s = latent_dict['z_camera']

        fusion_dict = {
            'sonar_dict': self.sonar_gan({'target': z_t, 'compl': z_s}, mode, stage),
            'camera_dict': self.camera_gan({'target': z_s, 'compl': z_t}, mode, stage)}

        return {
            'z': self.z_fuse(fusion_dict),
            'loss': self.get_loss(fusion_dict)
        }

class FinalModel(nn.Module):
    """docstring for GanFusionSingle: fusion module for one target modality"""
    def __init__(self, config, camera_model, sonar_model, num_classes):
        super(FinalModel, self).__init__()
        self.camera_model = camera_model
        self.camera_model.eval()
        self.sonar_model = sonar_model
        self.sonar_model.eval()
        self.gan_fusion = GanFusion(config)
        self.class_layer = nn.Linear(config['latent_dim'], num_classes)

    def forward(self,input_sonar, input_camera, mode, stage):
        sonar_embedding = self.sonar_model(input_sonar)
        camera_embedding = self.camera_model(input_camera)
        latent_dict = {
            'z_sonar': sonar_embedding,
            'z_camera': camera_embedding
        }

        gan_fusion_output = self.gan_fusion(latent_dict, mode, stage)
        class_output = self.class_layer(gan_fusion_output['z'])
        return gan_fusion_output, class_output

def train_GAN(model, camera_dataloader,sonar_dataloader, optimizer_gan,criterion_gan):
    model.gan_fusion.train()
    # Progress Bar 
    batch_bar = tqdm(total=len(camera_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    
    num_correct = 0
    total_loss = 0
    dataloader=zip(camera_dataloader,sonar_dataloader)
    losses = {'disc_real':0, 'disc_fake':0, 'gen':0}

    for i, data in enumerate(dataloader):
        
        image_camera=data[0][0]
        labels=data[0][1] #Same labels for both 
        image_sonar=data[1][0]
        # images=torch.concat([image_camera,image_sonar],dim=1)
        
        

        image_camera, image_sonar, labels = image_camera.to(device), image_sonar.to(device), labels.to(device)
        
        # for stage in ['disc_fake']:

        
        for stage in ['disc_real', 'disc_fake', 'gen']:
            optimizer_gan.zero_grad() # Zero gradients
            output_dict, class_output = model(image_sonar, image_camera, 'train', stage)
            loss_class = criterion_gan(class_output, labels)
            if(stage!='gen'):
                loss = output_dict['loss']
            else:
                loss= output_dict['loss'] + loss_class
            loss.backward()
            optimizer_gan.step()
            losses[stage] += output_dict['loss'].item()

            # num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        
        total_loss += float(loss_class.item())
        
        num_correct += int((torch.argmax(class_output, axis=1) == labels).sum())
        # total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (8*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.04f}".format(float(optimizer_gan.param_groups[0]['lr'])),
            disc_real="{:.04f}".format(float(losses['disc_real'] / (i + 1))),
            disc_fake="{:.04f}".format(float(losses['disc_fake'] / (i + 1))),
            gen="{:.04f}".format(float(losses['gen'] / (i + 1))))

            
        
        
        batch_bar.update()

    
    batch_bar.close()
    print("Training : ")
    print("Accuracy:","{:.04f}%".format(100 * num_correct / (8*(i + 1))))
    print("Classification Loss","{:.04f}".format(float(total_loss / (i + 1))))
    print("Discriminator Real","{:.04f}".format(float(losses['disc_real'] / (i + 1))))
    print("Discriminator Fake","{:.04f}".format(float(losses['disc_fake'] / (i + 1))))
    print("Generator","{:.04f}".format(float(losses['gen'] / (i + 1))))

    total_loss = float(total_loss / len(camera_dataloader))

    return total_loss

def valid_GAN(model, camera_dataloader,sonar_dataloader):
    model.eval()
    # Progress Bar 
    batch_bar = tqdm(total=len(camera_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Valid', ncols=5) 
    
    num_correct = 0
    total_loss = 0
    dataloader=zip(camera_dataloader,sonar_dataloader)
    losses = {'disc_real':0, 'disc_fake':0, 'gen':0}

    for i, data in enumerate(dataloader):
        
        image_camera=data[0][0]
        labels=data[0][1] #Same labels for both 
        image_sonar=data[1][0]
        # images=torch.concat([image_camera,image_sonar],dim=1)
        
        

        image_camera, image_sonar, labels = image_camera.to(device), image_sonar.to(device), labels.to(device)
        
        # for stage in ['disc_fake']:

        with torch.no_grad():
        
            for stage in ['disc_real', 'disc_fake', 'gen']:
                output_dict, class_output = model(image_sonar, image_camera, 'train', stage)
                loss_class = criterion_gan(class_output, labels)
                if(stage!='gen'):
                    loss = output_dict['loss']
                else:
                    loss= output_dict['loss'] + loss_class
                losses[stage] += output_dict['loss'].item()
            
            total_loss += float(loss_class.item())
        
        num_correct += int((torch.argmax(class_output, axis=1) == labels).sum())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (8*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            disc_real="{:.04f}".format(float(losses['disc_real'] / (i + 1))),
            disc_fake="{:.04f}".format(float(losses['disc_fake'] / (i + 1))),
            gen="{:.04f}".format(float(losses['gen'] / (i + 1))))
            
        
        
        batch_bar.update()

    batch_bar.close()
    print("Validation: ")
    print("Accuracy:","{:.04f}%".format(100 * num_correct / (8*(i + 1))))
    print("Classification Loss","{:.04f}".format(float(total_loss / (i + 1))))
    print("Discriminator Real","{:.04f}".format(float(losses['disc_real'] / (i + 1))))
    print("Discriminator Fake","{:.04f}".format(float(losses['disc_fake'] / (i + 1))))
    print("Generator","{:.04f}".format(float(losses['gen'] / (i + 1))))

    total_loss = float(total_loss / len(camera_dataloader))

    return total_loss

def train_ganfusion_model(train_loader_camera,train_loader_sonar,val_loader_camera,val_loader_sonar,camera_model,sonar_model):
    camera_model_fuse = torch.nn.Sequential(*(list(camera_model.children())[:-1]))
    sonar_model_fuse = torch.nn.Sequential(*(list(sonar_model.children())[:-1]))
    ganfusion_model = FinalModel(config, camera_model_fuse, sonar_model_fuse, 5).to(device)
    optimizer_gan =  torch.optim.Adam(ganfusion_model.parameters(), 1e-5)
    criterion_gan = nn.CrossEntropyLoss()

    for i in range(30):
        print('Epoch',i+1,'/30')
        train_GAN(ganfusion_model, train_loader_camera, train_loader_sonar, optimizer_gan,criterion_gan)
        valid_GAN(ganfusion_model, val_loader_camera, val_loader_sonar)

    return ganfusion_model



