# Speech Emotion Recognition Using LSTM

A deep learning project that recognizes human emotions from speech audio using **Librosa, MFCC feature extraction, audio augmentation, and a stacked Long Short-Term Memory (LSTM) neural network**. The model classifies speech into seven emotion categories and evaluates its performance using accuracy, loss, a confusion matrix, and a classification report.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features](#features)
- [Datasets](#datasets)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Accuracy and Performance](#accuracy-and-performance)
- [Test Results](#test-results)
- [Efficiency Analysis](#efficiency-analysis)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Outputs](#project-outputs)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Conclusion](#conclusion)

## Project Overview

Speech Emotion Recognition (SER) is a machine learning task that identifies emotion-related patterns in speech audio. Human speech contains acoustic information such as pitch, energy, and spectral characteristics that can help distinguish emotional expressions.

This project combines four emotion-labelled speech datasets, extracts acoustic features from audio recordings, applies audio augmentation, and trains a deep learning model to predict one of seven emotion classes.

The model is designed to classify audio into the following categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The project demonstrates an end-to-end machine learning workflow, including audio preprocessing, feature engineering, model training, and evaluation.

## Objectives

- Develop a deep learning model for speech emotion classification.
- Combine multiple publicly available speech emotion datasets.
- Extract meaningful acoustic features from speech recordings.
- Increase training diversity through audio augmentation.
- Evaluate classification performance using standard machine learning metrics.
- Analyze model efficiency and identify opportunities for improvement.

## Features

- **Multi-dataset learning:** combines TESS, RAVDESS, CREMA-D, and SAVEE.
- **Audio preprocessing:** loads speech recordings and processes audio segments using Librosa.
- **Feature extraction:** calculates MFCCs, Zero-Crossing Rate, and RMS energy.
- **Audio augmentation:** generates noisy and pitch-shifted versions of recordings.
- **Feature normalization:** applies StandardScaler to standardize input features.
- **Deep learning classification:** uses a stacked LSTM architecture with dropout.
- **Model evaluation:** measures accuracy and loss and generates a confusion matrix and classification report.
- **Visualization:** explores waveforms, spectrograms, and training performance.

## Datasets

The project combines four speech emotion datasets.

| Dataset | Description |
|---|---|
| TESS | Toronto Emotional Speech Set containing emotion-labelled speech recordings. |
| RAVDESS | Ryerson Audio-Visual Database of Emotional Speech and Song. |
| CREMA-D | Crowd-sourced Emotional Multimodal Actors Dataset. |
| SAVEE | Surrey Audio-Visual Expressed Emotion dataset. |

The notebook initially identifies **12,162 audio file paths** and successfully generates **18,880 feature examples** in its recorded run. The actual number of files processed can vary depending on dataset availability and local file paths.

### Emotion Classes

The model predicts seven emotion categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

These categories are encoded numerically before training the neural network.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computation |
| Pandas | Dataset management and feature storage |
| Librosa | Audio loading and signal processing |
| Matplotlib | Data and training visualizations |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Data splitting, feature scaling, label encoding, and evaluation |
| TensorFlow / Keras | Deep learning model development and training |
| Jupyter Notebook | Interactive experimentation and analysis |

## Methodology

The project follows these main stages.

### 1. Dataset Collection

Audio recordings are collected from TESS, RAVDESS, CREMA-D, and SAVEE. The notebook extracts emotion labels from dataset folder names or filename conventions and combines the available records into a common dataframe.

In this project, we will be using the following datasets, which are popular in the research community for training emotion recognition models:

**Crowd-sourced Emotional Multimodal Actors Dataset (Crema-D)** :https://www.kaggle.com/datasets/ejlok1/cremad

**Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)** :https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

**Surrey Audio-Visual Expressed Emotion (Savee)** :https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

**Toronto Emotional Speech Set (Tess)** :https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

### 2. Audio Preprocessing

Audio recordings are loaded using Librosa. The feature extraction function processes audio segments of up to 2.5 seconds, beginning at an offset of 0.6 seconds.

Audio is processed at a common sampling rate where applicable, allowing consistent feature extraction across recordings.

### 3. Audio Augmentation

Audio augmentation increases the diversity of training examples by applying transformations to the original signal.

The notebook generates four feature representations for each successfully processed source recording:

- **Original audio:** preserves the original acoustic characteristics.
- **Noise augmentation:** introduces random Gaussian noise.
- **Pitch shifting:** changes the pitch of the recording.
- **Combined augmentation:** applies pitch shifting and noise together.

These transformations can help the model learn from variations in acoustic characteristics. However, augmented recordings must be assigned to the same data partition as their original source to prevent data leakage during evaluation.

### 4. Feature Extraction

The notebook extracts three types of acoustic features:

- Zero-Crossing Rate (ZCR)
- Root Mean Square (RMS) energy
- Mel-Frequency Cepstral Coefficients (MFCCs)

These features are concatenated into numerical vectors that are supplied to the classification model.

### 5. Feature Scaling

The notebook uses `StandardScaler` to standardize the features. The scaler is fitted on the training partition and applied to the other partitions.

This helps prevent features with larger numerical ranges from dominating the learning process.

### 6. Model Training

A stacked LSTM neural network is trained using the extracted features. Dropout layers are included to help reduce overfitting.

The model uses the Adam optimizer and categorical cross-entropy loss to learn the relationship between acoustic features and emotion labels.

### 7. Model Evaluation

The notebook evaluates the model using:

- Accuracy
- Categorical cross-entropy loss
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics provide an initial assessment of classification performance and the types of errors made by the model.

## Feature Extraction

### Zero-Crossing Rate (ZCR)

Zero-Crossing Rate measures how frequently the audio waveform changes sign within a frame. It provides information about signal characteristics and can help distinguish different types of speech sounds.

### Root Mean Square (RMS) Energy

RMS energy estimates the average energy of an audio frame. It provides information about the signal's energy and intensity, which can contribute to distinguishing emotional speech patterns.

### Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs represent the short-term spectral envelope of an audio signal using a perceptually motivated frequency scale. They capture useful characteristics of speech and are widely used in speech recognition and audio classification.

Combining these features provides the model with information about the signal's energy, waveform behavior, and spectral structure.

## Model Architecture

The notebook uses a stacked LSTM classifier with two LSTM layers, dropout regularization, and a seven-unit softmax output layer.

| Layer | Configuration | Parameters |
|---|---|---:|
| LSTM Layer 1 | 64 units, returns sequences | 624,896 |
| Dropout Layer 1 | Dropout rate: 0.2 | 0 |
| LSTM Layer 2 | 64 units | 33,024 |
| Dropout Layer 2 | Dropout rate: 0.2 | 0 |
| Dense Output Layer | 7 units, softmax activation | 455 |
| **Total** | | **658,375** |

### LSTM Layers

Long Short-Term Memory networks are designed to learn patterns in sequential data. The model contains two LSTM layers to learn nonlinear relationships in the extracted feature representation.

### Dropout Regularization

Dropout with a rate of 0.2 is applied after each LSTM layer. During training, dropout randomly disables a portion of activations to help reduce overfitting.

### Dense Output Layer

The final dense layer contains seven neurons, one for each emotion class. The softmax activation converts the output into class probabilities.

### Loss Function and Optimizer

The model uses categorical cross-entropy as its loss function and Adam as its optimizer. Accuracy is used as the primary evaluation metric during training.

**Architecture note:** the notebook reshapes the input into `(samples, 1, 2376)`, meaning each recording is represented as one timestep containing 2,376 features. Consequently, the network does not learn temporal progression across audio frames in the usual sequence-modeling sense. A frame-level input representation would be more appropriate for exploiting the temporal capabilities of an LSTM.

## Training Configuration

The recorded notebook uses the following training configuration:

| Parameter | Value |
|---|---|
| Maximum epochs | 50 |
| Batch size | 64 |
| Initial learning rate | 0.001 |
| Dropout rate | 0.2 |
| Training/test split | 80% / 20% |
| Validation split | 10% of the initial training partition |
| Split random state | 42 |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Main metric | Accuracy |
| Learning-rate reduction factor | 0.5 |
| Early-stopping patience | 5 epochs |

The training process uses early stopping and a learning-rate reduction callback. The notebook records training output through epoch 50.

The notebook creates a separate validation partition, but its `model.fit()` call passes the test partition as `validation_data`. This should be corrected before using the reported performance as a final benchmark.

## Accuracy and Performance

The following metrics are reported by the saved notebook output.

| Metric | Recorded Result |
|---|---:|
| Test accuracy | **93.38%** |
| Test loss | **0.24568** |
| Test examples | **3,776** |
| Total model parameters | **658,375** |
| Approximate model size reported by Keras | **2.51 MB** |
| Final recorded training accuracy | Approximately 96.87% |
| Final recorded validation accuracy | Approximately 93.38% |

### Accuracy

The model reports a test accuracy of **93.38%**, indicating that approximately 93 out of every 100 examples in the evaluated partition were classified correctly.

This result reflects the notebook's current experimental split. It should not be interpreted as a verified estimate of performance on entirely unseen speakers or real-world audio because the notebook uses the test set during training and may distribute augmented versions of the same recording across different partitions.

### Loss

The recorded test loss is approximately **0.24568**. Categorical cross-entropy measures the difference between the predicted class probabilities and the true class labels. Lower loss is generally preferable when comparing experiments under the same evaluation conditions.

### Training and Validation Performance

The recorded training accuracy is higher than the validation accuracy. This difference indicates a generalization gap and may be consistent with overfitting.

However, because the test partition is used as validation data in the recorded training call, a clean training-versus-validation comparison requires rerunning the experiment with the correct data partitions.

## Test Results

### Classification Report

The notebook prints the following class-wise results.

| Printed Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Neutral | 0.95 | 0.92 | 0.94 | 499 |
| Happy | 0.92 | 0.95 | 0.93 | 541 |
| Sad | 0.95 | 0.92 | 0.93 | 530 |
| Angry | 0.92 | 0.90 | 0.91 | 515 |
| Fear | 0.93 | 0.96 | 0.94 | 621 |
| Disgust | 0.94 | 0.93 | 0.94 | 521 |
| Surprise | 0.93 | 0.93 | 0.93 | 549 |
| **Overall accuracy** | | | **0.93** | **3,776** |
| Macro average | 0.93 | 0.93 | 0.93 | 3,776 |
| Weighted average | 0.93 | 0.93 | 0.93 | 3,776 |

**Important:** the notebook passes emotion names in an order that does not match the `LabelEncoder` class order. Therefore, the per-class metrics above are reproduced as printed, but their emotion labels may be incorrect. Recalculate the classification report using the encoder's actual class order before treating these figures as final.

### Confusion Matrix

The notebook records the following confusion matrix. Rows represent true labels and columns represent predicted labels.

The order is `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

```text
[[461, 12,  4,  7,  7,  2,  6],
 [  6,515,  1,  5,  5,  6,  3],
 [  9,  6,489, 12,  1,  4,  9],
 [  5,  6,  9,466, 10,  3, 16],
 [  1,  4,  2,  3,595, 15,  1],
 [  0, 16,  4,  2, 11,487,  1],
 [  4,  3,  8, 10, 10,  1,513]]
```

The diagonal entries represent correct predictions, while the off-diagonal entries represent misclassifications.

The matrix can help identify emotion pairs that the model confuses. Such analysis should be repeated after correcting the validation protocol and class-label mapping.

## Efficiency Analysis

### Inference Performance

The saved notebook output indicates that evaluation and prediction on 3,776 examples each take approximately one second in the recorded environment.

These are environment-specific measurements rather than controlled benchmarks. Hardware, batch size, TensorFlow configuration, and runtime overhead can affect the results.

The notebook does not provide a controlled end-to-end latency benchmark that includes audio loading, feature extraction, preprocessing, and prediction.

### Model Complexity

The model contains 658,375 parameters and has an approximate size of 2.51 MB according to the recorded Keras model summary.

The first LSTM layer accounts for most of the model's parameters. Because the input contains a single timestep, a simpler dense neural network may achieve comparable performance with less architectural complexity.

### Efficiency Improvements

Potential improvements include:

- Benchmarking audio preprocessing and neural network inference separately.
- Comparing the LSTM with a smaller dense neural network.
- Using frame-level sequences if temporal modeling is required.
- Saving the trained model and fitted preprocessing objects for reuse.
- Measuring single-recording latency separately from batch throughput.
- Profiling CPU and GPU performance before claiming real-time capability.

## Installation

### Prerequisites

- Python 3.10 or another Python version compatible with the selected TensorFlow release.
- Jupyter Notebook or JupyterLab.
- The required Python libraries.
- Local copies of the datasets used by the notebook.

### Install Dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas librosa matplotlib seaborn scikit-learn tensorflow ipython jupyter
```

Exact dependency versions were not recorded in the notebook. For reproducible experiments, test the environment and pin the versions that work.

## How to Run

1. Clone or download the repository containing the notebook.
2. Download and arrange the TESS, RAVDESS, CREMA-D, and SAVEE datasets.
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Update the absolute dataset paths in the notebook to match your environment.
5. Run the notebook cells in order.
6. Allow time for audio preprocessing, feature extraction, augmentation, and model training.
7. Review the training history, final evaluation, confusion matrix, and classification report.
8. Correct the evaluation issues described below and rerun the experiment before publishing final benchmark claims.

The notebook writes `data_path.csv` and `features.csv` to the current working directory.

## Project Outputs

The notebook generates or displays the following outputs:

| Output | Description |
|---|---|
| `data_path.csv` | Combined emotion labels and audio file paths |
| `features.csv` | Extracted numerical features and emotion labels |
| Waveform plots | Visual representations of audio amplitude over time |
| Spectrograms | Visualizations of frequency content over time |
| Training plots | Accuracy and loss during training |
| Confusion matrix | Comparison of true and predicted emotion classes |
| Classification report | Precision, recall, F1-score, and support |

The saved notebook does not demonstrate a deployment-ready model export or saved preprocessing objects.

## Limitations and Future Improvements

### 1. Correct the Test and Validation Split

The notebook uses the test set as validation data during training. This allows test performance to influence training decisions.

**Improvement:** use the dedicated validation partition for early stopping and learning-rate scheduling. Reserve the test set for final evaluation only.

### 2. Prevent Augmentation Leakage

The notebook generates augmented feature rows before performing a random train/test split. Variants of the same original recording may appear in both partitions.

**Improvement:** split the original recordings, preferably by speaker, before augmentation. Apply augmentation only to the training partition.

### 3. Correct Emotion Label Mapping

The classification report uses a class-name order that differs from the fitted label encoder.

**Improvement:** use `lb.classes_` when generating the classification report and ensure the confusion-matrix labels use the same order.

### 4. Improve Temporal Modeling

The current model receives a single timestep containing the flattened feature vector. This limits its ability to learn temporal changes in speech.

**Improvement:** retain frame-level features as a time sequence, or compare the current architecture with a dense classifier better suited to fixed-length feature vectors.

### 5. Investigate Feature Dimensions

The example feature shapes in the notebook sum to 2,354 values, while the saved feature table and model input contain 2,376 features.

**Improvement:** investigate the discrepancy, enforce consistent feature dimensions, and document the final feature-extraction configuration.

### 6. Improve Reproducibility

The notebook does not establish a fully reproducible environment with pinned package versions and fixed random seeds for all augmentation and training operations.

**Improvement:** seed Python, NumPy, and TensorFlow; record dependency versions; and save the model, scaler, label encoder, and experiment configuration.

### 7. Evaluate Real-World Generalization

Performance on acted emotion datasets may differ from performance on spontaneous speech, different accents, unseen speakers, and noisy recordings.

**Improvement:** test on speaker-independent splits and an external dataset. Report per-class metrics with verified label mappings and document the limitations of the dataset annotations.

### 8. Save the Trained Model

The notebook does not demonstrate exporting the model and preprocessing objects for independent inference.

**Improvement:** save the trained model and the fitted scaler and label encoder. Ensure the inference pipeline applies the exact same preprocessing used during training.

## Conclusion

This project implements an end-to-end Speech Emotion Recognition pipeline using audio preprocessing, augmentation, MFCC-based feature extraction, and a stacked LSTM classifier.

The recorded notebook reports **93.38% accuracy**, a test loss of approximately **0.24568**, and **658,375 model parameters** across seven emotion classes.

These results demonstrate a promising baseline on the notebook's current evaluation split. However, the use of test data during training, potential augmentation leakage, inconsistent class-label ordering, and single-timestep LSTM input should be addressed before the results are presented as a reliable benchmark.

The next stage is to establish a speaker-independent evaluation, correct the label mapping, compare alternative architectures, and measure reproducible end-to-end inference performance.

---

**Note:** All reported metrics describe the saved notebook run. They should be updated if the preprocessing pipeline, dataset split, model architecture, or evaluation procedure changes.
