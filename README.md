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
## Project Structure

The repository is organized as follows:

```plaintext
├── Code/
│   ├── Enhancing_EEG_Signal_Reconstruction_in_Cross_Domain_Adaptation_Using_CycleGAN.ipynb
│
├── Dataset/
│   ├── trainA/         # Training data from domain A (MRI images)
│   ├── trainB/         # Training data from domain B (CWT images)
│   ├── testA/          # Testing data from domain A (MRI images)
│   ├── testB/          # Testing data from domain B (CWT images)
│
├── Images/
|  ├── Figure-10.png    # Diagram: Enhanced Model (MRI to CWT) with identity mapping loss
|  ├── Figure-11.png    # Diagram: Enhanced Model (CWT to MRI) with identity mapping loss
|  ├── Figure-13.png    # Diagram: Enhanced Model summary with all components
|  ├── Figure-8.png     # Diagram: Baseline Model (MRI to CWT) with cycle-consistency and adversarial losses
|  ├── Figure-9.png     # Diagram: Baseline Model (CWT to MRI) with cycle-consistency and adversarial losses
|
├── LICENSE             # License information for the project
├── README.md           # Main project description and usage instructions
├── requirements.txt    # Python dependencies required for the project

```

- **Code/:** Contains the main Jupyter notebook for the project.
- **Dataset/:** Contains preprocessed MRI and CWT images for training and testing.
  - **trainA:** MRI images for training.
  - **trainB:** CWT images for training.
  - **testA:** MRI images for testing.
  - **testB:** CWT images for testing.
- **Images/:** Contains all the flow diagrams illustrating the CycleGAN architecture and model processes:
  - **Figure-8.png:** Flow diagram of the baseline CycleGAN model for converting MRI to CWT images. Includes cycle-consistency and adversarial losses.
  - **Figure-9.png:** Flow diagram of the baseline CycleGAN model for converting CWT to MRI images. Includes cycle-consistency and adversarial losses.
  - **Figure-10.png:** Flow diagram of the Enhanced Model for converting MRI to CWT images with identity mapping loss.
  - **Figure-11.png:** Flow diagram of the Enhanced Model for converting CWT to MRI images with identity mapping loss.
  - **Figure-13.png:** Comprehensive diagram summarizing the Enhanced Model’s architecture with all components.
- **LICENSE:** A file specifying the license for this project (e.g., Creative Commons Attribution 4.0 International). It defines how the project can be used, modified, and distributed.
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

The Baseline Model follows the standard CycleGAN approach. It includes the typical adversarial loss and cycle consistency loss.

#### Parameters:
- **Lambda Identity ($\lambda_{\text{identity}}$):** 0
- **Adversarial Loss:** L1
- **Cycle Consistency Loss:** L1


#### Flow Diagram:

1. **MRI to CWT:** The following diagram shows how the MRI images are converted into CWT (Continuous Wavelet Transform) images by the generator.
    ![MRI to CWT](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/blob/main/Images/Figure-8.png)
  *Figure 1: Flow diagram representing the architecture of the baseline CycleGAN model for converting Magnetic Resonance (MRI) images to Continuous Wavelet Transform (CWT) images. The diagram illustrates the cycle-consistency loss and adversarial losses for both discriminators.*

3. **CWT to MRI:** The next diagram shows the reverse transformation, where CWT images are converted back to MRI images.
  ![CWT to MRI](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/blob/main/Images/Figure-9.png)
  *Figure 2: Flow diagram representing the process of converting Continuous Wavelet Transform (CWT) images back to Magnetic Resonance (MRI) images using the CycleGAN model. The diagram illustrates the cycle-consistency loss and adversarial losses for both discriminators.*
---

### 2. Adversarial Loss Modification

In the Modified Model, the adversarial loss function is replaced from L1 loss to Mean Squared Error (MSE) to enhance the stability and accuracy of the model. The flow diagrams of this model are the same as the Baseline Model and are depicted in Figures 1 and 2.

The parameters of this model are the same as the Baseline Model, except for the change in the adversarial loss. Specifically, the adversarial loss is now MSE-based.

---

### 3. Enhanced Model with Identity Loss

The Enhanced Model introduces identity mapping loss to preserve the structure of the input image. This loss ensures that the generator retains important features during translation, especially when an image is already in the target domain.

#### Parameters:
- **Lambda Identity ($\lambda_{\text{identity}}$):** 10
- **Identity Loss:** L1

#### Flow Diagram:
1. **MRI to CWT:** The transformation from MRI images to CWT images in the Enhanced model.
![MRI to CWT](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/blob/main/Images/Figure-10.png)
*Figure 3: Flow diagram representing the architecture of the CycleGAN model with identity mapping loss for converting Magnetic Resonance (MRI) images to Continuous Wavelet Transform (CWT) images. The diagram illustrates the cycle-consistency loss, adversarial losses, and identity mapping loss for both discriminators.*

2. **CWT to MRI:** The reverse transformation from CWT to MRI in the Enhanced model.
![CWT to MRI](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/blob/main/Images/Figure-11.png)
*Figure 4: Flow diagram representing the architecture of the CycleGAN model with identity mapping loss for converting Continuous Wavelet Transform (CWT) images back to Magnetic Resonance (MRI) images. The diagram illustrates the cycle-consistency loss, adversarial losses, and identity mapping loss for both discriminators.*
---

### Summary of Differences Between the Models:

| Model              | Loss Function                            | Key Feature                                      |
|--------------------|------------------------------------------|--------------------------------------------------|
| **Baseline Model**  | $L_{GAN}(G, D) + \lambda L_{cyc}(G, F)$  | Standard adversarial loss and cycle consistency. |
| **Modified Model**  | $L_{GAN}(G, D) + \lambda L_{cyc}(G, F)$ with MSE | MSE-based adversarial loss for smoother training. |
| **Enhanced Model**  | $L_{GAN}(G, D) + \lambda L_{cyc}(G, F) + \lambda_{identity} L_{identity}(G, F)$ | Adds identity loss to preserve features. |

---

## Requirements
Below are the libraries required for this project and their versions as of May 2024. These are included in the requirements.txt file:
```bash
albumentations==1.3.0
cv2==4.7.0
google-colab==1.0.0
matplotlib==3.7.2
mne==1.6.2
nibabel==5.1.0
numpy==1.24.3
pandas==1.5.3
Pillow==9.4.0
pywavelets==1.4.1
scipy==1.11.0
torch==2.0.1
torchvision==0.15.2
tqdm==4.64.1
pytorch-msssim==0.2.1
```

To install all dependencies in Google Colab, run the following command in a cell:
```bash
!pip install -r requirements.txt
```

## Results

### Quantitative Evaluation:

| Model                       | SSIM (MRI → CWT) | PSNR (MRI → CWT) | SSIM (CWT → MRI) | PSNR (CWT → MRI) |
|-----------------------------|------------------|------------------|------------------|------------------|
| **Baseline Model**          | 0.092            | 52.33            | 0.076            | 51.47            |
| **Modified Model**          | 0.215            | 54.85            | 0.092            | 52.60            |
| **Enhanced Model**          | 0.314            | 54.97            | 0.132            | 52.68            |

### Visual Comparison:
![Visual Comparison](https://github.com/Sameer-Ahmed7/EEG-Signal-Reconstruction-CycleGAN/blob/main/Images/Figure-13.png)
*Figure 5: Visual comparison of the generated images using the
three models. The figure shows the original input images and
the corresponding translated images generated by the Baseline
Model (Model 1), the model with Adversarial Loss Modification
(Model 2), and the model with Enhanced Loss With Identity
(Model 3). The comparison illustrates the performance and ac-
curacy of image translation according to different CycleGAN
configurations. "Odd rows represent (MRI to CWT), while even
rows represent (CWT to MRI)".*

## Installation and Usage

### Prerequisites:
- Python 3.8 or above  
- Google Colab (recommended)  
- Required libraries (see `requirements.txt`)  

### Running the Notebook:
1. Open the notebook on Google Colab using the link above.
2. Use the preprocessed dataset from the repository.
3. Follow the steps in the notebook for training and evaluation.

## Limitations
- **Domain Differences:** Structural (MRI) vs temporal (EEG) data integration poses challenges.
- **Signal Noise:** Artifacts in EEG signals can degrade the CWT quality.
- **CycleGAN Limitations:** Requires additional fine-tuning for biomedical data.

## Citation

If you use this repository, please cite the following paper:

```bibtex
@inproceedings{sameer2024eeg,
  title={Enhancing EEG Signal Reconstruction in Cross-Domain Adaptation Using CycleGAN},
  author={Samuele Russo, Sameer Ahmed, Imad Eddine TIBERMACINE and Christian Napoli},
  booktitle={2024 International Conference on Telecommunications and Intelligent Systems (ICTIS)},
  year={2024},
  organization={IEEE}
}
```

## License
This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). See the LICENSE file for details.
