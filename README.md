# 🎭 Real-Time Emotion Detector

![Demo GIF of the Emotion Detector](demo.gif)

A Python script that uses your webcam to perform real-time facial emotion recognition. It detects faces and displays the dominant emotion, along with a dynamic, semi-transparent dashboard showing the probabilities for all detected emotions.

---

## 🚀 Features

* **Real-Time Face Detection:** Uses OpenCV's built-in Haar Cascade to find faces in the webcam feed.
* **Deep Learning Emotion Analysis:** Employs the `DeepFace` library to analyze facial expressions and classify them into seven categories (happy, sad, angry, surprise, neutral, fear, disgust).
* **Dynamic Bounding Box:** Draws a rectangle around the detected face, colored according to the dominant emotion.
* **Emotion Dashboard:** Displays a clean, semi-transparent overlay with a sorted bar chart of all emotion probabilities, making it easy to see the full analysis.
* **Optimized Performance:** The script only runs the heavy `DeepFace.analyze` function once every 10 frames, ensuring a smooth video framerate.

---

## 🛠️ Technologies Used

* **Python 3**
* **OpenCV** (`opencv-python`): For webcam access, face detection, and drawing on the frame.
* **DeepFace**: For the core emotion recognition model.
* **NumPy**: For numerical operations.

---

## ⚙️ Setup and Usage

### 1. Clone the Repository

First, get the code on your local machine.
```bash
git clone [https://github.com/bhavya-tripathy/ComputerVisionProjects.git](https://github.com/bhavya-tripathy/ComputerVisionProjects.git)
cd ComputerVisionProjects/Your-Emotion-Project-Folder
