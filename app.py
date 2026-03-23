import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("DeepFakeImageDetection.h5")

st.title("DeepFake Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", width="stretch")

    # 🔥 IMPORTANT FIX STARTS HERE

    # Convert to RGB (safe)
    image = image.convert("RGB")

    # Resize (same as training)
    img = image.resize((224, 224))

    # Convert to numpy
    img = np.array(img)

    # ⚠️ FIX: RGB → BGR (agar OpenCV se train kiya tha)
    img = img[:, :, ::-1]

    # Normalize
    img = img / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # 🔥 FIX ENDS HERE

    # Prediction
    prediction = model.predict(img)

    # Debug (optional)
    st.write("Raw Prediction:", prediction)

    # Output (try both if unsure)
    if prediction[0][0] > 0.5:
        st.success("✅ Real Image")
    else:
        st.error("❌ Fake Image")