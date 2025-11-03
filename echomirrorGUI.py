import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
import pyttsx3
from deepface import DeepFace

try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Could not initialize TTS engine: {e}")
    engine = None

def speak(text):
    """ Function to speak the given text in a separate thread. """
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in speak function: {e}")
    else:
        print(f"TTS Engine not available. Would speak: {text}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load Haar cascade.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 0
last_spoken_emotion = None
last_spoken_time = 0
cooldown = 10 
dominant_emotion = "Unknown"


class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        
        self.frame_count = 0
        self.last_spoken_emotion = None
        self.last_spoken_time = 0
        self.cooldown = 10 

        self.title("Echomirror")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue")
        self.video_label = ctk.CTkLabel(self, text="Loading Camera...")
        self.video_label.pack(pady=20, padx=20, fill="both", expand=True)
        self.emotion_label = ctk.CTkLabel(self, text="Emotion: ---", font=ctk.CTkFont(size=20, weight="bold"))
        self.emotion_label.pack(pady=10)
        self.update_frame()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        global dominant_emotion
        ret, frame = cap.read()
        if not ret:
            self.video_label.configure(text="Camera Error")
            return
        
        frame = cv2.flip(frame, 1)
        if self.frame_count % 10 == 0:
            try:
                
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                if isinstance(analysis, list) and len(analysis) > 0:
                    emotion_predictions = analysis[0]['emotion']
                    dominant_emotion = analysis[0]['dominant_emotion'].lower()

                    if dominant_emotion == "happy":
                        current_time = time.time()
                        if (self.last_spoken_emotion != "happy") or (current_time - self.last_spoken_time > self.cooldown):
                            self.last_spoken_emotion = "happy"
                            self.last_spoken_time = current_time
                            
                            
                            speech_thread = threading.Thread(target=speak, args=("You look happy today, Bhavya. Keep that spark alive.",))
                            speech_thread.start()
                    else:
                        self.last_spoken_emotion = dominant_emotion
                
                else:
                    dominant_emotion = "Unknown"

            except Exception as e:
                
                dominant_emotion = "Error"

        emotion_map = {
            "happy": "😊",
            "sad": "😢",
            "neutral": "😐",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😮",
            "disgust": "🤢",
            "unknown": "...",
            "error": "❌"
        }
        emoji = emotion_map.get(dominant_emotion, "🤔")
        self.emotion_label.configure(text=f"Emotion: {dominant_emotion.capitalize()} {emoji}")
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        ctk_image = ctk.CTkImage(pil_image, size=(640, 480)) # Resize as needed
        
        self.video_label.configure(image=ctk_image, text="")
        self.frame_count += 1
        self.after(20, self.update_frame)

    def on_closing(self):
        print("Releasing resources...")
        cap.release()
        self.destroy()

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()