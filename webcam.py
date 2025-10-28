import cv2
from deepface import DeepFace


print("🎭 Starting Emotion Detector...")
print("Press 'Q' to quit anytime.")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


colors = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 0),
    "neutral": (200, 200, 200),
    "fear": (128, 0, 128),
    "disgust": (0, 128, 128)
}

frame_count = 0
emotion = "Detecting..."


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
                emotion = analysis[0]['dominant_emotion']
            else:
                emotion = "Unknown"
        except Exception as e:
            print(f"⚠️ Error during analysis: {e}")
            emotion = "Unknown"


    for (x, y, w, h) in faces:
        color = colors.get(emotion.lower(), (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion.capitalize(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('🎭 Emotion Detector', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
