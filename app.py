import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model('emotion_model.h5')

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Title on the front
st.title("🎭 Emotion Detection from Faces")
st.write("Upload a photo of your face, and we'll try to identify the emotion.😄😢😠😮")

uploaded_file = st.file_uploader("Upload a photo ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='The image that was uploaded', use_column_width=True)

    img = img.resize((48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Check if the predicted class is in the emotion_labels dictionary
    if predicted_class in emotion_labels:
        emotion = emotion_labels[predicted_class]
        st.subheader("🎯 Expectation:")
        st.success(f"Feelings: **{emotion}**")
    else:
        st.subheader("🎯 Expectation:")
        st.warning("Could not determine emotion.")
