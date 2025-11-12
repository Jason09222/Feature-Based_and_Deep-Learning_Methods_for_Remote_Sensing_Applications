# COMP9517 Group Project 

## More details can be found in [9517_modify.pdf](https://github.com/user-attachments/files/23493263/9517_modify.pdf)

This project uses the ​**Skyview Multi-Landscape Aerial Imagery Dataset** ([available on Kaggle](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset/data)), which contains aerial images of diverse landscapes. For classification, we implemented ​**2 traditional machine learning methods**  and ​**3 deep learning approaches** , leveraging both feature engineering and end-to-end training strategies.


## Requirements

This project is developed in the Google Colab environment and relies on the following Python libraries:

- **Core Libraries**:
  - Python 3.10.12  
  - NumPy 1.24.3  
  - OpenCV (cv2) 4.8.1.78  
  - Matplotlib 3.7.1  
  - tqdm 4.66.1  
  - Pandas 1.5.3  

- **Machine Learning**:
  - scikit-learn 1.3.0 (used for KNN, SVM, GridSearchCV, and evaluation metrics)

- **Deep Learning**:
  - PyTorch 2.1.0+cu121  
  - torchvision 0.16.0+cu121  
  - efficientnet-pytorch 0.7.1 (for EfficientNet-B0 implementation)

### Installation (for local use):
`pip install numpy opencv-python matplotlib scikit-learn tqdm pandas torch torchvision efficientnet-pytorch`

## Model Description

This project evaluates five different models, including both traditional computer vision approaches and deep learning architectures:

1. **LBP + KNN**  
   - Uses Local Binary Pattern (LBP) for feature extraction.  
   - K-Nearest Neighbors (KNN) is used as the classifier.  
   - Simple yet interpretable, but limited in feature representation.

2. **SIFT + SVM**  
   - SIFT descriptors capture local features.  
   - Support Vector Machine (SVM) is used for classification.  
   - More robust than LBP, especially for textured images.

3. **ResNet**  
   - Deep Convolutional Neural Network with residual connections.  
   - Strong baseline for image classification tasks.  
   - Provides significant performance boost compared to traditional methods.

4. **EfficientNet-B0**  
   - Lightweight and highly efficient CNN architecture.  
   - Achieves state-of-the-art performance with fewer parameters.  
   - Best performing model in this project.

5. **SENet**  
   - Squeeze-and-Excitation Network adds channel attention to standard CNNs.  
   - Slightly better than ResNet, but not as efficient as EfficientNet.

### Model Performance Comparison

| **Model**        | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|------------------|--------------|----------------|------------|-------------|
| LBP + KNN        | 55.08%       | 56.14%         | 55.08%     | 54.65%      |
| SIFT + SVM       | 67.12%       | 67.13%         | 67.12%     | 66.96%      |
| ResNet           | 95.08%       | 95.25%         | 95.08%     | 95.08%      |
| EfficientNet-B0  | **98.00%**   | **98.05%**     | **98.00%** | **98.00%**  |
| SENet            | 95.50%       | 95.55%         | 95.50%     | 95.50%      |

Note the training and test time are benchmarked using a NVIDIA Tesla T4 GPU.

Training Process of the above listed models:
The three deep learning models were trained for 10 epochs using the following data augmentation techniques for the training set： random horizontal flip, random rotation (up to 15°), and color jitter (brightness, contrast, saturation, hue). The images were then converted to tensors and normalized using ImageNet statistics.

### Introduction of the files

The code for the five models is contained in the following notebooks:
- **Model 1**: `LBP_KNN.ipynb`  
- **Model 2**: `SIFT_SVM.ipynb`  
- **Model 3**: `ResNet.ipynb`  
- **Model 4**: `EfficientNet.ipynb`  
- **Model 5**: `SENet.ipynb`  

Additionally, the following notebooks are used for specific tasks related to model interpretability and robustness:

- **`EfficientNet_GradCam.ipynb`**: This notebook demonstrates the implementation of **Grad-CAM** for visualizing the regions of an image that contribute most to the model's decision.
  
- **`EfficientNet_GradCam_Adversarial_Attacks.ipynb`**: This notebook explores the **interpretability of the model under adversarial attacks**, such as noise, blurring, or occlusion, and how these factors impact the model's predictions and visualizations.

### How to run the code

The code is designed to run on Google Colab with files stored in Google Drive. Follow these steps to run the notebooks:

1. **Mount Google Drive**:  
   First, mount your Google Drive to access the dataset and notebooks.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
2. **Copy the project folder**  
   After mounting your drive, copy the project folder to the Colab environment.

   ```bash
   !cp -r /content/drive/MyDrive/COMP9517_ZXCZH /content/
3. **Execute the notebooks**
Once the project folder is copied, open the individual notebooks and execute the code in the given order to obtain the results. In some training processes, we designed to generate loss graphs for easy observation, so we retained some of the images
