import cv2
from deepface import DeepFace
import numpy as np
import pyttsx3
import time
import threading  

print("🎭 Starting EchoMirror...")
print("Press 'Q' to quit anytime.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
colors = { 
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 0),
    "neutral": (200, 200, 200),
    "fear": (128, 0, 128),
    "disgust": (0, 128, 128)
}

def draw_emotion_dashboard(frame, predictions, colors):
    start_x, start_y = 10, 20
    bar_max_width = 150
    bar_height = 15
    text_offset = 75
    overlay_x, overlay_y = 5, 5
    overlay_w = text_offset + bar_max_width + 20
    overlay_h = (bar_height + 10) * len(predictions) + 15
    overlay = frame.copy()

    cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + overlay_w, overlay_y + overlay_h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    sorted_preds = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    for i, (emotion, prob) in enumerate(sorted_preds):
        bar_width = int((prob / 100) * bar_max_width)
        color = colors.get(emotion, (255, 255, 255))
        text_y_pos = start_y + i * (bar_height + 10) + bar_height // 2 + 5
        cv2.putText(frame, f"{emotion.capitalize()}:", (start_x, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (start_x + text_offset, start_y + i * (bar_height + 10)),
                      (start_x + text_offset + bar_width, start_y + i * (bar_height + 10) + bar_height),
                      color, -1)

engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 0.9)
voices = engine.getProperty('voices')

if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
else:
    engine.setProperty('voice', voices[0].id)

def speak(text_to_speak):
    """Runs the text-to-speech in a non-blocking thread."""
    try:
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speech thread: {e}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0
dominant_emotion = "Detecting..."
emotion_predictions = {}

last_spoken_emotion = None
last_spoken_time = 0
cooldown = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if frame_count % 10 == 0:  
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list) and len(analysis) > 0:
                emotion_predictions = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion'].lower()

                if dominant_emotion == "happy":
                    current_time = time.time()
                    if (last_spoken_emotion != "happy") or (current_time - last_spoken_time > cooldown):
                        last_spoken_emotion = "happy"
                        last_spoken_time = current_time
                        
                        speech_thread = threading.Thread(target=speak, args=("You look happy today, Bhavya. Keep that spark alive.",))
                        speech_thread.start()
                else:
                    last_spoken_emotion = dominant_emotion

            else:
                dominant_emotion = "Unknown"
                emotion_predictions = {}

        except Exception:
            dominant_emotion = "Unknown"
            emotion_predictions = {}

    for (x, y, w, h) in faces:
        color = colors.get(dominant_emotion, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = dominant_emotion.capitalize()
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y - 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 5, y - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    if emotion_predictions:
        draw_emotion_dashboard(frame, emotion_predictions, colors)

    cv2.imshow('EchoMirror', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()