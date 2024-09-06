# Audio File Classification

This project presents an approach for classifying `.wav` audio files based on features such as the intent of the speaker. The aim is to correctly identify and label audio files by analyzing their content, which is divided into two elements: **action** and **object**. The project applies two different machine learning approaches to solve this classification problem.

## Table of Contents

- [Problem Overview](#problem-overview)
- [Data](#data)
- [Research Approach](#research-approach)
- [Model Training](#model-training)
- [Results](#results)
- [Discussion and Future Work](#discussion-and-future-work)
- [References](#references)

## Problem Overview

The project addresses a classification problem where audio files need to be labeled based on the intent of the speaker. Each recording varies in length and includes various actions and objects (e.g., "decrease volume"). The data set is imbalanced, and the varying lengths of audio files pose a challenge. To solve this:

- **Development set**: 9,855 recordings.
- **Evaluation set**: 1,455 recordings.

We preprocess the data to balance the classes and extract relevant features for classification.

## Data

Each audio file in the dataset is accompanied by metadata (e.g., gender, age range). The audio files are preprocessed as follows:

- **Trimming**: Removal of leading and trailing silence.
- **Padding**: Standardizing audio length for input consistency.
- **Feature Extraction**: Extract Mel-Frequency Cepstral Coefficients (MFCC) to represent the spectral characteristics of the audio signal.

Additionally, the dataset is imbalanced, so we employ **oversampling** to ensure equal representation of classes.

## Research Approach

The pipeline for the project consists of:

### A. Data Preprocessing
- **Trimming and Padding**: Audio files are trimmed and padded to have a consistent input size.
- **MFCC Extraction**: We extract MFCCs, which are numerical coefficients representing spectral features of the signal.
- **Oversampling**: Minority classes are oversampled to balance the dataset.
- **Standardization**: Both training and test data are standardized to ensure that each feature has the same scale.

### B. Model Selection
Two different approaches were tested:

1. **Approach 1: Artificial Neural Network (ANN)**  
   - The standardized MFCCs are fed into a multi-layer fully connected ANN.
   - ANN parameters:
     - 4 hidden layers with neuron sizes: 1024, 512, 256, 128
     - Output layer with 7 nodes (for 7 classes).
   
2. **Approach 2: Random Forest and K-Nearest Neighbor (KNN)**  
   - MFCC features were segmented, and their mean and standard deviation were computed.
   - These were then used as input for Random Forest and KNN classifiers.
   - **Random Forest**: 90 trees, **KNN**: K=1.

### C. Hyperparameter Tuning
- Grid search was used for tuning the models:
  - Best number of K for KNN: 1.
  - Best number of trees in Random Forest: 90.
  
## Model Training

The following classifiers were used to train the model:
- **Artificial Neural Network (ANN)**
- **Random Forest (RF)**
- **K-Nearest Neighbor (KNN)**

The ANN performed the best in terms of accuracy.

## Results

The results from the two approaches are as follows:

| Model              | Accuracy |
|--------------------|----------|
| **ANN**            | 0.842    |
| **KNN (K=1)**      | 0.384    |
| **Random Forest**  | 0.412    |

The **ANN** model outperformed the other approaches, achieving the highest accuracy.

## Discussion and Future Work

To further improve the performance, several enhancements could be made:

- **Convolutional Neural Networks (CNNs)**: These networks can improve feature extraction from raw audio data.
- **Recurrent Neural Networks (RNNs)**: Specifically, Long Short-Term Memory (LSTM) networks can handle variable-length inputs and may provide more accurate predictions.
- Experimenting with other architectures like CNN-RNN hybrids could yield better performance in future work.

## References

1. Oo, M. M. (2018). Comparative study of MFCC feature with different machine learning techniques in acoustic scene classification. *International Journal of Research and Engineering, 5*, 439-444.
2. Cutler, A., Cutler, D. R., & Stevens, J. R. (2012). Random forests. *Ensemble Machine Learning: Methods and Applications*, 157-175.
3. Jain, A. K., & Dubes, R. C. (1988). Algorithms for clustering data. *Prentice-Hall, Inc.*
4. Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural networks for music classification. In *ICASSP 2017*, 2392-2396. IEEE.
5. Weninger, F., Erdogan, H., Watanabe, S., Vincent, E., Le Roux, J., & Hershey, J. R. (2015). Speech enhancement with LSTM recurrent neural networks and its application to noise-robust ASR. *LVA/ICA 2015*, 91-99. Springer International Publishing.
