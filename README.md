# Chest Pathologies Diagnosis

This project focuses on the diagnosis of 14 different chest pathologies using deep learning techniques. The model is trained on the CheXpert dataset, a well-known dataset in the field of medical imaging, which is not publicly accessible; users must obtain access before using it.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Chest pathologies can significantly impact patient health and require accurate diagnosis for effective treatment. This project aims to develop a robust model capable of identifying various chest conditions using advanced deep learning techniques.

## Dataset

The model is trained on the CheXpert dataset, which consists of chest X-ray images annotated for 14 different pathologies, including but not limited to:

1. Cardiomegaly
2. Edema
3. Consolidation
4. Pneumonia
5. Atelectasis
6. Pleural Effusion
7. Pneumothorax
8. Emphysema
9. Fibrosis
10. Mass
11. Nodule
12. Hernia
13. Fracture
14. Other

**Note:** To access the CheXpert dataset, you must request access from the official source.

## Methodology

This project employs transfer learning, leveraging the DenseNet architecture as the base model. Transfer learning allows the model to utilize pre-trained weights on large datasets, improving its ability to generalize from the smaller CheXpert dataset. 

### Steps:
1. **Data Preprocessing:** The images are resized and normalized for input into the DenseNet model.
2. **Model Training:** The model is trained on the processed dataset with appropriate hyperparameters to optimize performance.
3. **Evaluation:** The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

