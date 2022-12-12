11785: Intro to Deep Learning Project
Project Title: Sonar and Camera Fusion
Team Number: 22
Team Members AndrewIDs: anande, jezemba, mmangipu, mvisvana


Our project's experiments are all detailed in the IDL_Project.ipynb file.But for the sake of modularity, all the models can be run by simply running main.py with the command
        python main.py

Description of each file is given below

main.py - file which can run all the experiments and models 

data_loading.py - file which loads the data from the images and applies transformations

baselines.py - file consisting of unimodal baseline camera and sonar models

baseline_fusion.py - file which performs the simplest fusion technique by simply concatenating the camera and sonar embeddings

autofusion_model.py - file which performs autofusion on the trained sonar and camera embeddings

ganfusion_model.py - file which performs ganfusion on the trained sonar and camera embeddings


********************
IMPORTANT NOTE
Our code for AutoFusion and GANFusion was heavily inspired by the paper at https://arxiv.org/abs/1911.03821 and their github repository at https://github.com/Demfier/philo
********************


