🧠 Deep Learning on Fashion MNIST Dataset
This project implements a simple deep learning model using TensorFlow/Keras to classify images from the Fashion MNIST dataset. The model is trained to recognize clothing items across 10 different categories from grayscale images.

🎯 Objective
Build a basic fully connected neural network (DNN)

Train the model on Fashion MNIST images

Evaluate performance and visualize results

📦 Dataset
The Fashion MNIST dataset is used:

60,000 training images

10,000 testing images

Each image: 28×28 grayscale

10 classes:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

🗂️ File Structure & Code Explanation
The project is implemented in the following notebook:

Copy
Edit
📄 Deep_Learning.ipynb
Key notebook sections include:

1. 📚 Importing Libraries
tensorflow, numpy, matplotlib

2. 📥 Loading & Preprocessing Data
Loads Fashion MNIST via tf.keras.datasets

Normalizes pixel values to range [0, 1]

3. 🧠 Building the Model
A simple Sequential model with:

Flatten input layer

Dense hidden layer with ReLU

Dense output layer with softmax (10 units)

4. 🏋️ Model Compilation & Training
Loss: sparse_categorical_crossentropy

Optimizer: adam

Metric: accuracy

Trained using .fit() with validation

5. 📈 Evaluation & Visualization
Evaluates test accuracy

Visualizes sample predictions with class names

⚙️ Installation & Usage
📋 Prerequisites
bash
Copy
Edit
pip install tensorflow matplotlib numpy
▶️ Running the Notebook
Open the notebook in Jupyter or Google Colab.

Run all cells step by step:

Load dataset

Train the model

Evaluate and visualize results

👨‍💻 Author
Alaa Shorbaji
Artificial Intelligence Instructor
Deep Learning & Education Advocate

📜 License
This project is licensed under the MIT License.

✅ Use and adapt the code freely

❗ Provide attribution when sharing or publishing

❗ Include the license in any redistributions

Disclaimer: The dataset is provided by Zalando Research and available via TensorFlow Datasets.
