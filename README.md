# Iranian Gender Recognition

This repository contains a project focused on gender recognition for Iranian individuals using computer vision techniques. The aim is to provide a reliable, culturally relevant gender recognition model by leveraging facial features.

### Project Overview
The project utilizes two primary methods for face detection and gender classification:

1. **Haar Cascade Classifier** - A machine learning object detection approach that relies on Haar-like features to detect facial structures. This method is lightweight and efficient, making it suitable for real-time applications.
2. **Single Shot Detector (SSD)** - A deep learning-based approach that enables faster and more accurate face detection by identifying objects (in this case, faces) in a single forward pass of the network.

### Key Features
- **Data Preprocessing**: Efficient handling and preparation of images to align with model requirements.
- **Gender Classification**: Models are fine-tuned to recognize gender with cultural specificity for Iranian faces.
- **Real-time Detection**: Utilizing SSD for real-time performance while maintaining accuracy.

### Technologies Used
- **OpenCV**: For image processing and applying Haar Cascade for initial face detection.
- **TensorFlow/Keras**: To implement and train the SSD model for accurate detection and classification.
  
This project is designed to be extensible, allowing for additional tuning and improvement in accuracy. Contributions are welcome!

