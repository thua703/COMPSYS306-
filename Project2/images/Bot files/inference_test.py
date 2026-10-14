import pickle
import numpy as np
import cv2
import pibot


MODEL_PATH = "svm_final_model.pkl"


def preprocess_frame(frame):
    # pibot.camWait() returns BGR image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to same size used during training
    frame_resized = cv2.resize(
        frame_rgb,
        (64, 48),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float32 and normalize
    frame_array = frame_resized.astype(np.float32) / 255.0

    # Flatten: 64 x 48 x 3 = 9216 features
    features = frame_array.flatten()

    # SVC expects shape: (number_of_samples, number_of_features)
    features = features.reshape(1, -1)

    return features


# Load trained SVM
print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded.")


try:
    # Start camera
    pibot.camInit()

    print("Camera started.")
    print("Running inference... Press Ctrl+C to stop.")

    while True:
        # Wait for a new camera frame
        frame = pibot.camWait()

        # Preprocess
        features = preprocess_frame(frame)

        # Predict
        prediction = model.predict(features)[0]

        print("Prediction:", prediction)


except KeyboardInterrupt:
    print("\nStopping...")


finally:
    pibot.camDeinit()
    print("Camera stopped.")