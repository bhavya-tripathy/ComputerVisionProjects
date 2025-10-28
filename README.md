# 🎭 Real-Time Emotion Detector  
> *"See what your face says before you do."*

![Demo GIF](demo.gif)

A real-time **facial emotion recognition system** built with **OpenCV** and **DeepFace**.  
This project detects human faces from your webcam feed, analyzes their emotions using deep learning, and displays a smooth, semi-transparent emotion dashboard that updates dynamically with every expression.

---

## 🚀 Features

- **🎥 Real-Time Face Detection:**  
  Detects faces instantly using OpenCV’s Haar Cascade classifier.

- **🧠 Deep Learning Emotion Analysis:**  
  Powered by **DeepFace**, classifying seven emotions —  
  `Happy`, `Sad`, `Angry`, `Surprise`, `Neutral`, `Fear`, and `Disgust`.

- **🌈 Emotion-Based Visualization:**  
  Draws dynamic bounding boxes around faces, color-coded by dominant emotion.

- **📊 Emotion Dashboard:**  
  A minimal, semi-transparent overlay showing sorted emotion probabilities as bars.

- **⚡ Optimized Performance:**  
  Runs emotion analysis only once every 10 frames, balancing accuracy and real-time speed.

---

## 🛠️ Tech Stack

| Technology | Role |
|-------------|------|
| 🐍 Python 3 | Core language |
| 🎥 OpenCV | Webcam access, face detection, frame rendering |
| 🤖 DeepFace | Pre-trained emotion recognition model |
| 🔢 NumPy | Numerical operations and performance optimization |

---

## ⚙️ Setup & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/bhavya-tripathy/ComputerVisionProjects.git
cd ComputerVisionProjects/Real-Time-Emotion-Detector
