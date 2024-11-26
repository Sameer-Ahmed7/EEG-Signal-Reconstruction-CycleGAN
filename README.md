# Enhancing EEG Signal Reconstruction in Cross-Domain Adaptation Using CycleGAN


This repository contains the implementation of our research paper, accepted at the **2024 International Conference on Telecommunications and Intelligent Systems (ICTIS)**.

### **Paper Title:**  
**"Enhancing EEG Signal Reconstruction in Cross-Domain Adaptation Using CycleGAN"**

### **Authors:**  
- Samuele Russo  
- Sameer Ahmed  
- Imad Eddine Tibermacine  
- Prof. Christian Napoli  

---

## Project Description

This project demonstrates the use of Cycle-Consistent Generative Adversarial Networks (CycleGAN) to synthesize and enhance EEG signals from MRI data. The goal is to improve the quality of EEG signals for Brain-Computer Interfaces (BCI) by leveraging structural information from MRI scans.

### Key Features:
- Reconstruction of EEG signals from MRI data.  
- Application of three CycleGAN models with varying loss functions.  
- Evaluation metrics include **Structural Similarity Index Measure (SSIM)** and **Peak Signal-to-Noise Ratio (PSNR)**.  

---

## Repository Structure

```plaintext
├── code/
│   ├── Enhancing_EEG_Signal_Reconstruction_in_Cross_Domain_Adaptation_Using_CycleGAN.ipynb
│
├── dataset/
│   ├── trainA/         # Training data from domain A (MRI images)
│   ├── trainB/         # Training data from domain B (CWT images)
│   ├── testA/          # Testing data from domain A (MRI images)
│   ├── testB/          # Testing data from domain B (CWT images)
│
├── README.md           # Main project description
├── requirements.txt    # Python dependencies
├── LICENSE             # License information
```

- **Code/:** Contains the main Jupyter notebook for the project.
- **Dataset/:** Contains preprocessed MRI and CWT images for training and testing.
  - **trainA:** MRI images for training.
  - **trainB:** CWT images for training.
  - **testA:** MRI images for testing.
  - **testB:** CWT images for testing.
- **README.md:** Provides an overview of the project.
- **requirements.txt:** Lists required Python libraries.

## Google Colab Notebooks

The project is implemented in a single Jupyter notebook, which can be executed directly on Google Colab. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jvIsLSV2J_eJzUwcIOUV3I6p3IMAmbqm#scrollTo=lWGWzn4JYZbG)

### Steps to Run:
- Click the Colab badge above to open the notebook.
- Follow the instructions in the notebook.

## Dataset
### Options for Data Access:
You can directly use the preprocessed dataset provided in the Dataset/ folder of this repository.
- Preprocessed Dataset:
  - Use the preprocessed MRI and CWT images from the trainA, trainB, testA, and testB folders.
  - Skip the DATA DESCRIPTION AND PREPROCESSING step in the Colab notebook and go directly to the METHODOLOGY section.
  - Dataset Structure:
    - trainA/: MRI images for training.
    - trainB/: CWT images for training.
    - testA/: MRI images for testing.
    - testB/: CWT images for testing.
- Raw Dataset:
  - If you prefer to start with raw data, download the Real-World Table Tennis Dataset from OpenNeuro.
  - Upload the data to Google Drive and follow the DATA DESCRIPTION AND PREPROCESSING section in the notebook to prepare it for training.

## Methodology

### Overview of the CycleGAN Models

We developed three CycleGAN models for this project, each with unique loss functions and architectures:

1. **Baseline Model**: The simplest CycleGAN model with standard losses.
2. **Modified Model**: Introduces a modified adversarial loss to improve stability.
3. **Enhanced Model**: Adds identity loss to preserve input features during translation.

Each model uses a different approach to balance the generator's ability to create realistic images and the discriminator's ability to distinguish real from fake images.

---

### 1. Baseline Model

The initial and simplest model is a baseline architecture based on CycleGAN, adhering to the parameters outlined in the official CycleGAN paper. The flow diagram for this model is provided below}. Specifically, the baseline model adopts the original CycleGAN architecture with the following parameters:

- **Lambda Identity ($\lambda_{\text{identity}}$):** 0
- **Adversarial Loss:** L1
- **Cycle Consistency Loss:** L1


#### Flow Diagram:

The flow diagram for the **Baseline Model** is a standard CycleGAN generator and discriminator.

![Baseline Model Architecture](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/raw/main/images/baseline-architecture.png)

- **Generator**: Transforms images from domain X (MRI) to domain Y (CWT).
- **Discriminator**: Distinguishes between real and generated CWT images.

---

### 2. Adversarial Loss Modification

In Model 2, the adversarial loss function is replaced from an
L1 to a Mean Squared Error (MSE) to enhance the stability
and accuracy of the model. The flow diagram of this
model is the same as the baseline model which is depicted
in Figure 8, and Figure 9.The parameters are the same as the
baseline model except for the following:

- **Adversarial Loss:** MSE The reason to use MSE is
that it can give a more generalized gradient which
enhances the performance of adversarial network.

The cycle consistency loss does not need to be modified to
retain the consistency of mapping between the two domains.

#### Flow Diagram:

The flow diagram of the **Adversarial Loss Modification Model** is similar to the Baseline Model, but the discriminator is trained using MSE loss.

![Modified Model Architecture](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/raw/main/images/modified-architecture.png)

- **Generator**: Transforms MRI images to CWT, just like in the Baseline model.
- **Discriminator**: Uses MSE loss to assess the generated CWT images.

---

### 3. Enhanced Model with Identity Loss

In Model 3, identity loss is incorporated from Model 2 to
help the generators maintain the features of the image input
from the target domain. This model is represented as a flow
diagram as illustrated below. The
parameters are the same as Model 2 except for the following:

- **Lambda Identity ($\lambda_{\text{identity}}$):** 10
- **Identity Loss:** L1

#### Flow Diagram:

The flow diagram of the **Enhanced Model with Identity Loss** is similar to the previous models, but with the addition of the identity loss term.

![Enhanced Model Architecture](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/raw/main/images/enhanced-architecture.png)

- **Generator**: Transforms MRI to CWT while preserving important features through identity loss.
- **Discriminator**: Identifies whether images are real or generated, similar to the Baseline model, but enhanced by identity loss.

---

### Summary of Differences Between the Models:

| Model              | Loss Function                            | Key Feature                                      |
|--------------------|------------------------------------------|--------------------------------------------------|
| **Baseline Model**  | $L_{GAN}(G, D) + \lambda L_{cyc}(G, F)$  | Standard adversarial loss and cycle consistency. |
| **Modified Model**  | $L_{GAN}(G, D) + \lambda L_{cyc}(G, F)$ with MSE | MSE-based adversarial loss for smoother training. |
| **Enhanced Model**  | $L_{GAN}(G, D) + \lambda L_{cyc}(G, F) + \lambda_{identity} L_{identity}(G, F)$ | Adds identity loss to preserve features. |

---
