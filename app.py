import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gesture_segmentation import segment_hand

try:
    model = load_model('sign_language_model.h5')
except (IOError, ImportError):
    print("Model not found. Please train the model by running model.py first.")
    exit()


LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def main():
    """
    Main function to run the sign language detection application.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        roi_top, roi_bottom, roi_right, roi_left = 100, 300, 350, 550
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]

        cv2.rectangle(clone, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        segmented_hand = segment_hand(roi)

        if segmented_hand is not None:
            segmented_hand = cv2.resize(segmented_hand, (28, 28))
            prediction_image = np.expand_dims(np.expand_dims(segmented_hand, axis=0), axis=-1)

            
            prediction = model.predict(prediction_image)
            predicted_label = LABELS[np.argmax(prediction)]

            
            cv2.putText(clone, f"Prediction: {predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sign Language Detection", clone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
