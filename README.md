# Heart Disease Prediction Using Multi-Layer Perceptron (MLP)

This project aims to build an Artificial Neural Network (ANN) model capable of predicting the likelihood of heart disease in patients based on a set of clinical and biological indicators. The globally recognized Heart Disease Dataset was utilized to train and evaluate the model. The significance of this project lies in providing an intelligent clinical decision-support tool that can contribute to early diagnosis and intervention.

---

## 1. Dataset Description
The dataset contains various medical attributes of patients. The key features provided to the neural network include:
* **Age and Sex**
* **Chest Pain Type (cp)**
* **Resting Blood Pressure and Cholesterol (trestbps & chol)**
* **Target Variable:** A binary classification indicating health status (`0 = Healthy`, `1 = Heart Disease`).

### Data Preprocessing
* **Data Splitting:** The data was partitioned into 80% for training to allow the network to learn underlying patterns, and 20% for testing to evaluate its generalization performance.
* **Feature Scaling:** The `StandardScaler` algorithm was applied to normalize the numerical ranges of the features (Mean = 0, Variance = 1). This is a crucial step to prevent gradient bias and ensure stable convergence during Gradient Descent.

---

## 2. Model Architecture
A feedforward Multi-Layer Perceptron (MLP) neural network was constructed with the following architectural pipeline:

1. **Input Layer:** Receives the normalized clinical variables.
2. **First Hidden Layer:** Consists of 32 neurons utilizing the `ReLU` activation function to extract complex, non-linear relationships within the data.
3. **Dropout Layer (0.2):** Integrated to randomly drop 20% of the neurons during training, protecting the model against Overfitting.
4. **Second Hidden Layer:** Consists of 16 neurons with the `ReLU` activation function to enhance the network's representation capacity.
5. **Output Layer:** Contains a single neuron paired with the `Sigmoid` activation function to output a probability value between 0 and 1, perfectly matching the requirements of Binary Classification.

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam optimizer, selected for its efficiency and rapid convergence properties.

---

## 3. Results and Evaluation
Following the training phase, the model was evaluated on a completely unseen test set (61 samples) to measure its real-world performance. The network demonstrated highly stable and robust results, summarized from the Classification Report below:

### First: Global Metrics
* **Overall Accuracy:** The model achieved an overall accuracy of **87%**, reflecting an excellent capability to correctly classify healthy and diseased status across the test samples.

### Second: Per-Class Performance Analysis

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Healthy)** | 86% | 86% | 86% | 29 patients |
| **1 (Heart Disease)** | 88% | 88% | 88% | 32 patients |

* **Precision:** Reached **88%** for the disease class, meaning that when the model predicts a patient has heart disease, it is correct 88% of the time.
* **Recall:** Reached **88%** for the disease class, indicating that the network successfully identified 88% of all actual patients with heart disease in the dataset. This high sensitivity is critical in medical diagnostics to minimize dangerous False Negatives.
