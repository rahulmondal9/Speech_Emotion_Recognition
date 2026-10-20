Speech Emotion Recognition Using LSTM
A deep-learning project that classifies speech audio into seven emotion categories using acoustic features extracted from speech and a stacked Long Short-Term Memory (LSTM) neural network. This README is based on the supplied Speech_Emotion_Recognition(1).ipynb notebook and its saved outputs.
Reported result: the notebook records 93.38% test accuracy and 0.2457 test loss on 3,776 examples. This is the notebook's reported evaluation, not a guarantee of performance on new speakers or real-world audio. Important validation and label-reporting caveats are documented below.

Table of Contents
- Project Overview
- Goals and Features
- Datasets
- Pipeline and Feature Engineering
- Model Architecture
- Training Configuration
- Accuracy, Performance, and Test Results
- Efficiency
- Reproducibility and Setup
- How to Run
- Important Limitations and Recommended Fixes
- Project Outputs
- Conclusion
Project Overview
Speech Emotion Recognition (SER) aims to infer an emotion label from the acoustic characteristics of a spoken recording. This notebook combines audio from several emotion-recognition datasets, extracts numerical features from each clip, augments the audio to increase training examples, and trains a neural network to predict one of seven classes.

The model predicts one label per input feature vector. It does not transcribe speech, identify the words being spoken, or establish a person's actual internal emotional state. It learns patterns associated with emotion labels in the datasets.
Goals and Features
- Combine labelled audio from TESS, RAVDESS, CREMA-D, and SAVEE.
- Inspect audio using waveforms and spectrograms.
- Load clips with Librosa at a common sample rate.
- Extract Zero-Crossing Rate (ZCR), Root Mean Square (RMS) energy, and Mel-Frequency Cepstral Coefficients (MFCCs).
- Apply noise and pitch augmentation to increase feature examples.
- Standardize features and encode emotion labels.
- Train a stacked two-layer LSTM classifier with dropout and a softmax output.
- Evaluate using accuracy, loss, a confusion matrix, and a classification report.
Datasets
The notebook builds a combined dataframe from these four sources:
Dataset	Role in this notebook
TESS (Toronto Emotional Speech Set)	Emotion-labelled WAV recordings; ps is mapped to surprise.
RAVDESS	Emotion codes are mapped to labels; codes 1 and 2 are both mapped to neutral.
CREMA-D	Filename emotion codes such as SAD, ANG, DIS, FEA, HAP, and NEU are mapped to labels.
SAVEE	Filename codes are mapped to emotion labels.


The combined dataframe initially contains 12,162 file paths. The feature-generation loop successfully processes the paths that exist in the runtime environment and creates 18,880 feature rows. Since the notebook uses local Windows paths, the exact dataset composition and processed count depend on the folders and files available on the machine running it.
Emotion classes
The label encoder used for model training sorts the seven output classes as follows:
1. angry
2. disgust
3. fear
4. happy
5. neutral
6. sad
7. surprise
The model's final dense layer has seven softmax outputs, one per class.
Pipeline and Feature Engineering
1. Audio loading
Audio is loaded using librosa.load. Feature extraction uses clips up to 2.5 seconds, beginning at an offset of 0.6 seconds. The notebook shows a sample loaded at 22,050 Hz. Librosa's default resampling behavior is used when no sample rate is explicitly supplied.
2. Audio augmentation
The feature pipeline creates four representations per successfully processed source clip:
- Original audio — the unmodified crop.
- Noise-augmented audio — adds random Gaussian noise at a randomly selected amplitude.
- Pitch-shifted audio — applies a randomly selected pitch shift.
- Pitch-shifted plus noise — combines a random pitch shift and noise.
The notebook defines time-stretching and shifting helper functions for demonstrations, but the get_features() function used to build the final feature dataset does not call those two helpers. The augmentation strategy therefore consists of the original clip plus three augmented variants, not all four augmentation techniques shown in the exploratory cells.
3. Extracted features
The notebook concatenates these features into one flat vector:
- Zero-Crossing Rate (ZCR): counts how frequently the waveform changes sign within a frame. It can provide information about noisiness and some spectral characteristics.
- Root Mean Square (RMS) energy: estimates signal energy/loudness over short frames.
- MFCCs: represent the short-term spectral envelope on a perceptually motivated mel scale and are widely used as speech/audio features.
With the notebook's example settings, ZCR has 107 values, RMS has 107 values, and the flattened MFCC representation has 2,140 values. The resulting vector has 2,354 values by this arithmetic; however, the saved feature table and downstream arrays have 2,376 input columns. This discrepancy should be investigated before treating the pipeline as fully reproducible. The model was trained using the 2,376-column feature table present in the saved notebook output.
4. Missing-value handling
The notebook detects missing values in the feature dataframe and replaces them with zero using fillna(0). This prevents NaN values from entering the model, but it does not establish whether the missing values are harmless. A production pipeline should investigate why they occur and ensure that zero is an appropriate replacement.
5. Scaling and splitting
The feature matrix is split into a training portion and a held-out portion with test_size=0.2 and random_state=42. A further 10% of the initial training portion is set aside as validation data.
Partition	Rows	Approx. share of all feature rows
Training	13,593	72.0%
Validation	1,511	8.0%
Test	3,776	20.0%
Total	18,880	100%


StandardScaler is fitted on the training partition and then applied to validation and test partitions. This is the correct direction for avoiding scaler-fit leakage. The notebook then reshapes features for the LSTM.

Important: the split is performed after augmentation, and the augmented versions of one original recording may therefore be distributed across training and test sets. This can make the reported test score optimistic. For a stronger estimate of generalization, split by original recording—and preferably by speaker—before augmentation.
Model Architecture
The notebook defines a stacked LSTM classifier:
Layer	Configuration	Parameters
LSTM 1	64 units, returns sequences	624,896
Dropout 1	Rate 0.2	0
LSTM 2	64 units	33,024
Dropout 2	Rate 0.2	0
Dense output	7 units, softmax	455
Total		658,375


- Input shape used for training: (1, 2376) per example (one timestep containing 2,376 features).
- Output: seven softmax probabilities.
- Loss: categorical cross-entropy.
- Optimizer: Adam, with the compiled model using the default Adam learning rate (0.001 in the recorded training logs).
- Main metric: accuracy.
Architecture interpretation
LSTM networks are designed to learn patterns across sequences. In this notebook, the input is reshaped to one timestep with all 2,376 features at that timestep. Consequently, the network cannot learn temporal progression across audio frames in the usual sequence-modeling sense; it behaves more like a high-capacity nonlinear classifier over a single feature vector. A more sequence-oriented design would retain the frame/time dimension rather than flattening all MFCC, ZCR, and RMS values into one timestep.
Training Configuration
Setting	Value in notebook
Maximum epochs	50
Batch size	64
Initial learning rate in logs	0.001
Dropout	0.2 after each LSTM
Early stopping	Monitors val_acc, patience 5, restores best weights
Learning-rate reduction	Monitors val_acc, patience 3, factor 0.5, minimum learning rate 0.00001
Random seed for split	42


The recorded training output runs through epoch 50. The learning-rate callback is configured to reduce the learning rate when its monitored metric plateaus. Note that the notebook uses val_acc as the callback monitor while the visible Keras history keys are val_accuracy; depending on the Keras version, the monitor name may not match. Confirm the callback logs and use val_accuracy consistently for current TensorFlow/Keras versions.

There is also a validation-design issue: the model.fit() call passes X_test_reshaped, Y_test as validation_data, even though a separate validation partition was created. Thus, the test set is used to monitor training and learning-rate changes. The reported test score is not a fully untouched final test estimate. Use X_val_reshaped, Y_val for callbacks and reserve the test set for one final evaluation after all choices are fixed.
Accuracy, Performance, and Test Results
The following values are taken from the saved notebook outputs.
Overall evaluation
Metric	Recorded value	Interpretation
Test accuracy	93.3792% (reported as 93.38%)	Fraction of the 3,776 test examples classified correctly in the notebook evaluation.
Test loss	0.24568 (about 0.2457)	Mean categorical cross-entropy on the evaluated set; lower is better for comparable experiments.
Test examples	3,776	Number of examples in the held-out partition used by the notebook.
Model parameters	658,375	Trainable parameter count shown in the model summary.
Maximum epochs	50	The recorded fit output reaches epoch 50.


At 93.3792% accuracy on 3,776 examples, the result corresponds to approximately 3,525 correct and 251 incorrect predictions (the exact integer count should be verified from predictions if exact reporting is required). This is strong performance on the notebook's split, but the data leakage risks above mean it should not be interpreted as a reliable estimate of performance on unseen speakers or deployment audio.
Recorded class-wise report
The notebook prints the following classification report. Caution: its target_names are passed from emotion_names, whose order is neutral, happy, sad, angry, fear, disgust, surprise; the LabelEncoder output order is angry, disgust, fear, happy, neutral, sad, surprise. These orders do not match. Therefore, the class-wise precision/recall/F1 labels printed below may be misassigned even though the aggregate accuracy remains 93.38%. Recompute the report using lb.classes_ (the same encoder used to create Y) before publishing class-wise metrics.
Printed class name	Precision	Recall	F1-score	Support
neutral	0.95	0.92	0.94	499
happy	0.92	0.95	0.93	541
sad	0.95	0.92	0.93	530
angry	0.92	0.90	0.91	515
fear	0.93	0.96	0.94	621
disgust	0.94	0.93	0.94	521
surprise	0.93	0.93	0.93	549
Overall accuracy			0.93	3,776
Macro average	0.93	0.93	0.93	3,776
Weighted average	0.93	0.93	0.93	3,776


These class-level values are reproduced as printed for transparency, not endorsed as correctly mapped to each emotion. The confusion matrix uses the encoder order listed above, so its row/column labels should also use that order.
Confusion matrix
The saved notebook output contains this confusion matrix, with rows representing true labels and columns representing predicted labels in label-encoder order (angry, disgust, fear, happy, neutral, sad, surprise):
[[461, 12,  4,  7,  7,  2,  6],
 [  6,515,  1,  5,  5,  6,  3],
 [  9,  6,489, 12,  1,  4,  9],
 [  5,  6,  9,466, 10,  3, 16],
 [  1,  4,  2,  3,595, 15,  1],
 [  0, 16,  4,  2, 11,487,  1],
 [  4,  3,  8, 10, 10,  1,513]]
The diagonal entries are correct predictions; off-diagonal entries are confusions between classes. For example, the matrix shows 16 true happy examples predicted as surprise, and 16 true sad examples predicted as disgust in the encoder's class order. Investigate these confusions after correcting the evaluation protocol and label mapping.
Training behavior and generalization
The final recorded epoch reports approximately 96.87% training accuracy, 93.38% validation accuracy, training loss 0.0898, and validation loss 0.2457. The training accuracy is higher than the evaluation accuracy, indicating some generalization gap. This can be consistent with overfitting, though the data split and use of test data for validation also affect the interpretation. The notebook's separate final evaluation reports 93.38% accuracy and 0.24568 loss.
Efficiency
Observed inference speed
The notebook's evaluation and prediction outputs show approximately:
- model.evaluate() on 3,776 examples: about 1 second in the recorded environment.
- model.predict() on 3,776 examples: about 1 second in the recorded environment.
- The evaluation progress output shows roughly 4 ms/step for its batch steps.
These timings are environment-specific notebook timings, not controlled benchmarks. They include framework/runtime effects and should not be presented as a guaranteed per-clip latency. The notebook does not record a separate, controlled end-to-end benchmark for audio decoding, feature extraction, and prediction.
Computational footprint
- 658,375 parameters and a model summary size of approximately 2.51 MB are reported by Keras.
- Batch size is 64, which processes examples in mini-batches during training.
- The flattened 2,376-feature input is relatively wide, and the first LSTM layer accounts for most of the model parameters.
- Feature generation processes each valid recording and generates four feature rows (original plus three augmented variants), increasing preprocessing work and storage requirements.
Practical efficiency recommendations
1. Benchmark feature extraction and model inference separately, then measure total latency from WAV input to emotion prediction.
2. Save the fitted StandardScaler and label encoder with the model so inference does not repeat or change training preprocessing.
3. Compare the LSTM with a dense neural network or a smaller model: because the input has only one timestep, an LSTM may add complexity without exploiting temporal recurrence.
4. Consider retaining frame-level features for a genuine sequence model, or use a compact fixed-length feature vector with a model suited to tabular inputs.
5. Use batched prediction for throughput; measure single-clip latency separately if the intended application is interactive.
6. Profile CPU and GPU inference before claiming real-time capability. The notebook does not specify hardware, memory usage, model size on disk, or a standardized latency benchmark.
Reproducibility and Setup
The notebook imports these main libraries:
- Python standard library: os, sys, zipfile, pathlib, warnings
- Data handling: numpy, pandas
- Audio: librosa
- Visualization: matplotlib, seaborn
- Machine learning: scikit-learn
- Deep learning: TensorFlow / Keras
- Notebook audio playback: IPython.display.Audio
A typical environment can be created with:
python -m venv .venv
Activate the environment, then install the dependencies:
pip install numpy pandas librosa matplotlib seaborn scikit-learn tensorflow ipython jupyter
The notebook does not record exact package versions, so this is a practical dependency list rather than a locked, verified environment. For reproducible work, pin the tested versions in a requirements.txt after confirming the notebook runs cleanly.
Dataset paths must be updated
The notebook uses absolute Windows paths such as C:\Users\rahul\OneDrive\Desktop\Project\.... Update the Tess, Ravdess_zip_path, extracted_dir, Crema, and Savee variables to match your local dataset locations. Ensure the directory structure and filename conventions match the parsing logic. The notebook also writes data_path.csv and features.csv to the current working directory.
How to Run
1. Download and arrange the TESS, RAVDESS, CREMA-D, and SAVEE datasets in directories accessible to your Python environment.
2. Open Speech_Emotion_Recognition(1).ipynb in Jupyter Notebook or JupyterLab.
3. Update all absolute dataset paths in the dataset-loading cells.
4. Run the notebook from the first cell in order. Dataset loading and feature extraction can take time.
5. Check the generated data_path.csv and features.csv files and confirm that no unexpected paths were skipped.
6. Review the printed model summary, training logs, final evaluation, confusion matrix, and classification report.
7. Before treating the metrics as final, fix the validation/test split and class-label ordering issues described below, then rerun the experiment.
The notebook does not currently save the trained model or preprocessing objects as a deployment-ready artifact. Add explicit saving/loading code if the model will be used outside the notebook.
Important Limitations and Recommended Fixes
These issues are important to address before presenting the score as a robust benchmark or deploying the model.
1. Test set is used during training
The model.fit() call uses X_test_reshaped, Y_test as validation_data, even though a separate validation set is created. Early stopping and learning-rate scheduling can therefore use test performance to influence training.

Fix: use the actual validation arrays in model.fit() and evaluate on the test set only once after training decisions are complete.
2. Augmented variants can leak across splits
The notebook expands each original recording into multiple variants before randomly splitting feature rows. A variant of a recording may be in training while another variant of the same recording is in testing.

Fix: split source recordings (and ideally speakers) into train/validation/test groups first, then augment only the training group.
3. Classification-report label order is inconsistent
LabelEncoder class order differs from emotion_names, which is passed as the target_names argument to classification_report. This can attach the wrong emotion names to the printed per-class metrics.

Fix: generate the report with target_names=lb.classes_ and use the same order for confusion-matrix labels.
4. The LSTM input has one timestep
The data is reshaped to (samples, 1, 2376). This does not provide a temporal sequence of audio frames to the LSTM.

Fix: either provide properly structured frame-level sequences, or compare against a dense classifier and other suitable models using the flattened feature vector.
5. Feature-vector dimension discrepancy
The example printed in the feature-extraction cells shows ZCR (107,), RMS (107,), and MFCC (2140,), which sum to 2,354 values, whereas the generated dataframe and model inputs contain 2,376 features.

Fix: inspect actual feature lengths across clips, explicitly enforce a consistent shape, and document the intended dimension. The notebook's saved output alone does not explain the 22-feature difference.
6. Callback metric naming
Callbacks monitor val_acc, while modern Keras commonly records val_accuracy. Depending on the installed version, the callback may not monitor the intended metric.

Fix: use monitor="val_accuracy" consistently and confirm the training log shows the callbacks activating.
7. Reproducibility is incomplete
The split uses random_state=42, but augmentation uses random operations without a fixed seed, and framework-level random seeds are not set. Dependency versions and hardware are not recorded.

Fix: seed Python, NumPy, and TensorFlow; record package versions; save the scaler, label mapping, model, and experiment configuration.
8. Dataset and real-world limitations
Emotion labels are dataset annotations, and acted speech may differ from spontaneous speech. Recording conditions, language, accents, speaker demographics, microphones, and background noise can affect performance. Overall accuracy alone cannot establish fairness, calibration, robustness, or real-world reliability.

In this project, we will be using the following datasets, which are popular in the research community for training emotion recognition models:

**Crowd-sourced Emotional Multimodal Actors Dataset (Crema-D)** :https://www.kaggle.com/datasets/ejlok1/cremad

**Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)** :https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

**Surrey Audio-Visual Expressed Emotion (Savee)** :https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

**Toronto Emotional Speech Set (Tess)** :https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
Fix: evaluate on a held-out dataset or unseen speakers, report class-wise metrics with verified labels, and test realistic noisy audio before deployment.
Project Outputs
The notebook generates or displays:
- data_path.csv — combined emotion labels and audio file paths.
- features.csv — extracted feature vectors with emotion labels.
- Waveform and spectrogram visualizations.
- Training/validation accuracy and loss plots.
- A confusion matrix.
- A classification report.
The saved notebook does not show an exported model file or saved scaler/encoder.
Conclusion
The notebook demonstrates a complete baseline Speech Emotion Recognition workflow: dataset aggregation, audio augmentation, feature extraction, scaling, LSTM training, and evaluation. Its saved run reports 93.38% accuracy on 3,776 examples, with test loss 0.2457, a model size of approximately 2.51 MB, and 658,375 parameters.

The most important next step is not simply to increase model complexity. First, make the evaluation trustworthy by separating recordings/speakers before augmentation, using the validation set rather than the test set during training, and correcting the class-label order in the classification report. Then benchmark end-to-end latency and compare the LSTM with simpler models. Those changes will make accuracy, generalization, and efficiency claims substantially more defensible.
Suggested citation
If this project is used in a report or portfolio, describe the result as:
“The notebook's current experimental split achieved 93.38% accuracy on 3,776 feature examples. Because augmented variants may cross split boundaries and the test set was used as training validation data, this score should be treated as preliminary until a recording- or speaker-independent evaluation is performed.”


