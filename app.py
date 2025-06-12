import streamlit as st
import cv2
from fer import FER
import numpy as np

# Movie suggestions based on emotions
emotion_movies = {
    "happy": ["The Pursuit of Happyness", "Paddington 2", "Life is Beautiful", "La La Land","The Intouchables"],
    "sad": ["The Fault in Our Stars", "A Silent Voice", "Inside Out", "Manchester by the Sea","The Green Mile"],
    "angry": ["John Wick", "Gladiator", "The Dark Knight", "Kill Bill","12 Angry Men"],
    "surprise": ["Inception", "Interstellar", "The Prestige", "Now You See Me","The Prestige"],
    "fear": ["Get Out", "The Conjuring", "A Quiet Place", "Hereditary"],
    "disgust": ["The Platform", "Train to Busan", "Mother!", "Parasite","Psycho"],
    "neutral": ["Forrest Gump", "The Secret Life of Walter Mitty", "Chef", "Cast Away"]
}

# Streamlit UI
st.title("🎥 Emotion-Based Movie Recommender")
st.write("Take a photo and get a movie suggestion based on your facial expression.")

# Take snapshot
run = st.button("📸 Take a Snapshot")

if run:
    # Open webcam
    cap = cv2.VideoCapture(0)
    st.write("Starting camera... Please look at the camera.")
    
    if not cap.isOpened():
        st.error("Camera couldn't be opened.")
    else:
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Display image
            st.image(frame, channels="BGR", caption="Captured Image")

            # Detect emotion
            detector = FER(mtcnn=True)
            result = detector.detect_emotions(frame)

            if result:
                emotions = result[0]["emotions"]
                st.subheader("😃 Emotion Scores:")
                for emotion, score in emotions.items():
                    st.write(f"{emotion.capitalize()}: {round(score * 100, 2)}%")

                # Top emotion
                top_emotion = max(emotions, key=emotions.get)
                st.success(f"Top Emotion: **{top_emotion.capitalize()}**")

                # Suggest movies
                st.subheader("🍿 Recommended Movies:")
                for movie in emotion_movies.get(top_emotion, ["No suggestions found"]):
                    st.write(f"• {movie}")
            else:
                st.error("No face detected. Try again.")
        else:
            st.error("Couldn't capture image.")
