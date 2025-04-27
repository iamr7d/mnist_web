# MNIST Digit Recognition Web App

This project is a web application for real-time handwritten digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features
- Draw digits on a web canvas and get instant predictions
- Uses a high-accuracy pretrained Keras/TensorFlow model
- Flask backend for prediction API
- Modern, responsive web UI

## Getting Started

### Requirements
- Python 3.8+
- pip
- (Recommended) Virtual environment

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/iamr7d/mnist_web.git
   cd mnist_web
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download or train the model using `download_pretrained_mnist.py`.

### Running the App
```bash
python app.py
```
Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## File Structure
- `app.py` - Flask backend
- `templetes/index.html` - Web UI
- `MODEL/model.h5` - Pretrained Keras model
- `download_pretrained_mnist.py` - Script to download/train a strong MNIST model

## License
MIT
