

---

# 📚 CIFAR-10 Image Classification with CNN

This project builds and trains a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images from the **CIFAR-10 dataset**.

The CIFAR-10 dataset contains **60,000 images (32×32 pixels, RGB)** across **10 categories**:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

---

## 🧠 Model Architecture

```python
from tensorflow.keras import models, layers

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPool2D(pool_size=(2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

### 🔍 Layer Breakdown

* **Conv2D (32 filters, 3×3 kernel, ReLU)** → learns low-level features (edges, colors, textures).
* **MaxPooling2D (2×2)** → reduces image size (downsampling).
* **Conv2D (64 filters, 3×3, ReLU)** → learns higher-level features (shapes, object parts).
* **MaxPooling2D (2×2)** → further reduces size.
* **Flatten** → converts 2D feature maps into 1D vector.
* **Dense (64 neurons, ReLU)** → learns combinations of features.
* **Dense (10 neurons, Softmax)** → outputs class probabilities for 10 categories.

---

## ⚙️ Compilation & Training

```python
cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = cnn.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)
```

* **Optimizer:** Adam (adaptive learning rate).
* **Loss:** Sparse categorical cross-entropy (since labels are integers 0–9).
* **Metric:** Accuracy.

---

## 📊 Results

* Training for **100 epochs** gives \~**65% accuracy** on the CIFAR-10 test set.
* This is expected for a **simple CNN**.
* Accuracy can be improved with:

  * More convolutional layers (deeper network)
  * Dropout & Batch Normalization
  * Data Augmentation
  * Transfer Learning with pretrained models (e.g., ResNet, VGG16)

---

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install tensorflow matplotlib
   ```
2. Run the Python script:

   ```bash
   python cifar10_cnn.py
   ```
3. Training will start, and accuracy/loss will be shown for each epoch.

---

## 🛠️ Future Improvements

* Add **data augmentation** to prevent overfitting.
* Introduce **Batch Normalization & Dropout** layers.
* Use **transfer learning** with pre-trained ImageNet models for higher accuracy.
* Experiment with **learning rate schedules** and **optimizers**.

---

## 📈 Accuracy Trend (Sample)

| Epoch | Training Accuracy | Validation Accuracy |
| ----- | ----------------- | ------------------- |
| 1     | 45%               | 47%                 |
| 5     | 60%               | 62%                 |
| 10    | 65%               | 65%                 |

---

✨ **Result:** This simple CNN achieves \~65% accuracy on CIFAR-10, serving as a foundation for deeper experiments in computer vision.

---


