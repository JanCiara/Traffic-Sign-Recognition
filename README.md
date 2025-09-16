# Traffic Sign Recognition with Deep Learning 🚦

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of traffic signs. The model is trained to recognize 43 different categories of signs, demonstrating a practical application of deep learning for computer vision.

**Course:** [CS50's Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/2020/) - Harvard University

---

## 📝 Project Overview

The `traffic.py` script automates the entire machine learning pipeline, from data ingestion to model evaluation.

1.  **Data Loading and Preprocessing**
    - Loads thousands of images from the German Traffic Sign Recognition Benchmark (GTSRB) dataset using **OpenCV**.
    - Assumes a directory structure where each subdirectory is a unique class (e.g., `0/`, `1/`, `2/`).
    - Resizes all images to a uniform `30x30` pixel dimension with 3 color channels (RGB).

2.  **Data Splitting**
    - Splits the dataset into a training set and a testing set using **scikit-learn**.
    - A 60/40 split (`TEST_SIZE = 0.4`) is used to ensure the model is evaluated on data it has never seen before.

3.  **Model Building**
    - Constructs a powerful Convolutional Neural Network using the **TensorFlow** Keras API.
    - The architecture includes:
        - Two `Conv2D` layers to learn hierarchical visual features.
        - Two `MaxPooling2D` layers to reduce dimensionality and improve efficiency.
        - A `Flatten` layer to prepare data for the classification stage.
        - A `Dense` hidden layer with 512 neurons to learn complex patterns.
        - A `Dropout` layer (rate of 0.5) to prevent overfitting.
        - A final `Dense` output layer with a `softmax` activation function for multi-class classification.

4.  **Model Training**
    - Trains the CNN on the training data for a predefined number of `EPOCHS` (10).
    - Uses the `adam` optimizer and `categorical_crossentropy` loss function, which are standard for multi-class image classification.

5.  **Model Evaluation**
    - Evaluates the trained model's performance on the unseen test set.
    - Reports the final `loss` and `accuracy` to assess the model's real-world effectiveness.

6.  **Saving the Model (Optional)**
    - Provides a command-line option to save the trained model weights and architecture to an HDF5 (`.h5`) file for future use without retraining.

---

## 🛠️ Technologies Used

-   **Language:** Python 3
-   **Core Libraries:**
    -   [**TensorFlow**](https://www.tensorflow.org/): For building, training, and evaluating the deep learning model.
    -   [**scikit-learn**](https://scikit-learn.org/): For splitting the dataset into training and testing sets.
    -   [**OpenCV (cv2)**](https://opencv.org/): For high-performance image loading and manipulation.
    -   [**NumPy**](https://numpy.org/): For efficient numerical operations and handling image arrays.

---

## 📊 Dataset

The project is designed to work with the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

-   **Structure:** The dataset should be organized into directories, where each directory `0`, `1`, `2`, ..., `42` contains images corresponding to that traffic sign category.
-   **Features:** The raw pixel data of the `30x30x3` images.
-   **Target Variable:** The category of the traffic sign, represented by an integer from `0` to `42`.

---

## 🚀 How to Run the Project

### Prerequisites
-   Python 3
-   The GTSRB dataset (or a similarly structured image dataset).
-   Required Python libraries.

### Installation & Execution

1.  **Clone the repository** (if applicable)
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install dependencies**
    ```bash
    pip install tensorflow scikit-learn opencv-python numpy
    ```

3.  **Run the script to train the model**
    Provide the path to your data directory.
    ```bash
    python traffic.py gtsrb
    ```

4.  **Run and save the trained model**
    To save the resulting model, provide a second command-line argument.
    ```bash
    python traffic.py gtsrb model.h5
    ```

### Example Output
```bash
Starting to load data from: gtsrb
Epoch 1/10
499/499 [==============================] - 5s 9ms/step - loss: 2.1264 - accuracy: 0.4682
Epoch 2/10
499/499 [==============================] - 4s 9ms/step - loss: 0.6552 - accuracy: 0.8037
...
Epoch 10/10
499/499 [==============================] - 4s 9ms/step - loss: 0.0553 - accuracy: 0.9839
333/333 - 1s - loss: 0.1384 - accuracy: 0.9678
Model saved to model.h5.
```
*Note: The exact accuracy and loss values may vary slightly each time you run the script.*

---

## 🙏 Acknowledgments

This project is an assignment from **CS50's Introduction to Artificial Intelligence with Python**. The problem specification and project structure were provided by the course staff at Harvard University.
