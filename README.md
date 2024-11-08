**Chest X-ray Classification for Respiratory Conditions Using Hybrid CNN-SVM

Huey Hoang, Gaoming Lin

Project Description:
This project aims to develop a hybrid CNN-SVM model for diagnosing respiratory conditions based on Chest X-ray images. The idea is inspired by a paper about the vanilla convolutional neural network for COVID-19 detection [1]. In the hybrid CNN-SVM model, CNN works as an automatic feature extractor and SVM works as a binary classifier [2]. We want to use a hybrid CNN-SVM model to improve the work in this work. Our model will tell between normal individuals and patients with different respiratory conditions such as Pneumonia (COVID-19), Severe Acute Respiratory Syndrome (SARS), Streptococcus, and Acute Respiratory Distress Syndrome (ARDS).

Dataset:
The dataset is a collection of Chest X-rays of Healthy vs Pneumonia (Corona) affected patients along with other categories shown such as severe acute respiratory syndrome, streptococcus, and acute respiratory distress syndrome [3].

Pattern Recognition Techniques:
Data Preprocess: Removing noise and inconsistencies in the dataset to ensure high-quality input.
Data Visualization/Exploration: Understand data distribution and identify patterns through visualization tools such as matplotlib.
Feature Selection/Extraction: Extract deep features from images using Convolutional Neural Networks (CNNs).
Classification Method: Traditional ML classifiers like Support Vector Machines (SVM) and Random Forests (RF) trained on CNN-extracted features will be used for final classification. To create a convolution neural network, we will use libraries such as PyTorch. To create a support vector machine/random forest, we can use scikit-learn.
Model Selection: Employing cross-validation and hyperparameter tuning to select the best-performing models.
Error Estimation: Evaluating model performance using accuracy, precision, recall, F1-score, and ROC-AUC metrics.
References

[1] A. Dumakude and A. E. Ezugwu, “Automated COVID-19 detection with convolutional neural networks,” Scientific Reports, vol. 13, no. 1, p. 10607, Jun. 2023, doi: https://doi.org/10.1038/s41598-023-37743-4.

[2] S. Ahlawat and A. Choudhary, “Hybrid CNN-SVM Classifier for Handwritten Digit Recognition,” Procedia Computer Science, vol. 167, pp. 2554-2560, 2020, doi: https://doi.org/10.1016/j.procs.2020.03.309.

[3] Praveen, “CoronaHack -Chest X-Ray-Dataset,” Kaggle.com, 2019. https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_Metadata.csv (accessed Nov. 02, 2024).



