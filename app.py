import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Breast Mammogram Classifier", layout="centered")

CLASS_NAMES = ['BENIGN', 'MALIGNANT'] 

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/home/student/Breast_Cancer/breast_cancer_classifier.keras')
    
model = load_model()

st.title("Breast Mammogram Classifier")
st.write(
    "This tool classifies a cropped mammography image region as **Benign** or "
    "**Malignant**. It works on breast mammogram images only."
)
st.warning(
    "⚠️ **This is a student coursework prototype, not a medical device.** "
    "It has not been clinically validated. Testing showed it performs close to a "
    "naive baseline and currently misses a majority of malignant cases — do not use "
    "this for any real medical decision."
)

uploaded_file = st.file_uploader("Upload a mammogram image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)

    predicted_idx = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(prediction[0][predicted_idx]) * 100

    st.subheader("Result")
    st.write(f"**Predicted class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.1f}%")

    if confidence < 60:
        st.info(
            "The model is not very confident about this prediction. Treat this "
            "result as unreliable rather than a real finding."
        )

    st.write(
        f"In plain terms: the model predicts this image most resembles a "
        f"**{predicted_class.lower()}** region, with {confidence:.0f}% confidence. "
        f"This is not a diagnosis — only a qualified radiologist can determine that."
    )

st.markdown("---")
with st.expander("About this app"):
    st.write(
        """
        **How to run this locally:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Make sure `breast_cancer_classifier.keras` is in the same folder as this file
        3. Run: `streamlit run app.py`

        **Model:** small custom CNN trained from scratch on the CBIS-DDSM dataset
        (mammography, benign vs malignant classification). Coursework prototype only.
        """
    )