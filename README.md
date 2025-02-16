# Sonar-Camera Fusion Using Deep Learning

## Overview

This project demonstrates the fusion of sonar and camera data using deep learning techniques. The goal is to leverage the complementary nature of sonar and camera sensors to improve the accuracy of object detection and classification tasks.

## Features

- **Data Loading and Preprocessing**: Efficiently load and preprocess sonar and camera data.
- **Baseline Models**: Train and evaluate individual sonar and camera models.
- **Fusion Models**: Implement and train models that fuse sonar and camera data using various techniques:
  - Concatenated Embeddings
  - AutoFusion
  - GAN Fusion

## Project Structure

- `data_loading.py`: Contains functions for loading and preprocessing data.
- `baselines.py`: Contains baseline models for sonar and camera data.
- `baseline_fusion.py`: Contains the model for concatenated embeddings fusion.
- `autofusion_model.py`: Contains the AutoFusion model.
- `gan_fusion_model.py`: Contains the GAN Fusion model.
- `main.py`: Main script to run the training and evaluation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Sonar-Camera-Fusion-Using-Deep-Learning.git
    cd Sonar-Camera-Fusion-Using-Deep-Learning
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preparation**: Ensure your sonar and camera data is organized in the appropriate directories.

2. **Training Baseline Models**:
    ```bash
    python main.py --mode train_baseline
    ```

3. **Training Fusion Models**:
    ```bash
    python main.py --mode train_fusion
    ```

4. **Evaluating Models**:
    ```bash
    python main.py --mode evaluate
    ```

## Results

The project includes detailed results and visualizations of the performance of the baseline and fusion models. These results demonstrate the effectiveness of data fusion in improving classification accuracy.

## Acknowledgements

This project was developed as part of a deep learning course. Special thanks to the course instructors and peers for their valuable feedback and support.

********************
IMPORTANT NOTE
Our code for AutoFusion and GANFusion was heavily inspired by the paper at https://arxiv.org/abs/1911.03821 and their github repository at https://github.com/Demfier/philo
********************


