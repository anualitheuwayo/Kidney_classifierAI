
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = "kidney_ct_mobilenetv2_balanced_best.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]


st.set_page_config(
    page_title="KidneyVision AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2.3rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.8rem 2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0F3D56 0%, #176B87 100%);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            color: white;
            font-size: 2.2rem;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            font-size: 1.02rem;
            opacity: 0.92;
        }

        .section-label {
            color: #176B87;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid #D7E3EA;
            background-color: #F8FBFD;
            margin-top: 0.4rem;
        }

        .result-title {
            color: #46616D;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .prediction-value {
            color: #0F3D56;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0;
        }

        .small-note {
            color: #60727C;
            font-size: 0.88rem;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed #74A9C2;
            border-radius: 14px;
            padding: 0.8rem;
            background-color: #F8FBFD;
        }

        [data-testid="stMetric"] {
            background-color: #F8FBFD;
            border: 1px solid #D7E3EA;
            padding: 0.8rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource
def load_model(path: str):
    """Load and cache the trained Keras model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Place the .keras model in the same folder as app.py."
        )

    return tf.keras.models.load_model(path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and preprocess an uploaded image for MobileNetV2 inference."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)

    img_array = np.asarray(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


def predict_image(image: Image.Image, model):
    """Return prediction class, confidence, and all class probabilities."""
    img_tensor = preprocess_image(image)
    probabilities = model.predict(img_tensor, verbose=0)[0]

    prediction_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[prediction_index]
    confidence = float(probabilities[prediction_index])

    return predicted_class, confidence, probabilities



with st.sidebar:
    st.title("🩻 KidneyVision AI")

    st.divider()

    st.subheader("Model information")
    st.markdown(
        """
        - **Architecture:** MobileNetV2
        - **Input size:** 224 × 224 pixels
        - **Scan type:** Kidney CT
        - **Classes:** 4
        """
    )

    st.divider()

    st.subheader("Classification labels")
    st.markdown(
        """
        - Cyst
        - Normal
        - Stone
        - Tumor
        """
    )

    st.divider()

    st.info(
        "This application is for educational purposes only. "
        "It is not a medical device and must not be used for diagnosis or treatment decisions."
    )

st.markdown(
    """
    <div class="hero">
        <h1>Kidney CT Scan Classifier</h1>
        <p>
            Upload a kidney CT image to receive an AI-based four-class prediction:
            Cyst, Normal, Stone, or Tumor.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as error:
    st.error("Unable to start the classifier because the model file is missing.")
    st.code(str(error))
    st.stop()
except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.stop()


st.markdown('<p class="section-label">Step 1 · Upload image</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a kidney CT scan image",
    type=["png", "jpg", "jpeg"],
    help="Accepted file types: PNG, JPG, JPEG.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="small-note">
            Upload one CT image to begin analysis. The image will be resized to
            224 × 224 pixels before being sent to the MobileNetV2 model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")

    guide_col1, guide_col2, guide_col3 = st.columns(3)

    with guide_col1:
        st.info("**1. Upload**\n\nSelect one kidney CT image in PNG or JPG format.")

    with guide_col2:
        st.info("**2. Analyze**\n\nThe image is resized and normalized for the trained model.")

    with guide_col3:
        st.info("**3. Review**\n\nView the predicted class and probabilities for all labels.")

else:
    try:
        image = Image.open(uploaded_file)

        st.markdown('<p class="section-label">Step 2 · Image analysis</p>', unsafe_allow_html=True)

        left_col, right_col = st.columns([1.05, 1], gap="large")

        with left_col:
            st.subheader("Uploaded scan")
            st.image(
                image,
                caption=f"File: {uploaded_file.name}",
                use_container_width=True,
            )

            with st.expander("Image details"):
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Original dimensions:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Image mode:** {image.mode}")
                st.write("**Model input size:** 224 × 224 pixels")

        with right_col:
            st.subheader("Prediction result")

            with st.spinner("Running AI analysis..."):
                predicted_class, confidence, probabilities = predict_image(image, model)

            if confidence >= 0.80:
                confidence_label = "High model confidence"
                status_text = "The model assigned a comparatively strong probability to this class."
            elif confidence >= 0.50:
                confidence_label = "Moderate model confidence"
                status_text = "The model prediction should be interpreted cautiously."
            else:
                confidence_label = "Low model confidence"
                status_text = "The probabilities are relatively uncertain; do not rely on this output."

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Predicted category</div>
                    <div class="prediction-value">{predicted_class}</div>
                    <br>
                    <div class="result-title">Model confidence</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #176B87;">
                        {confidence:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(float(confidence))

            if confidence >= 0.80:
                st.success(f"**{confidence_label}:** {status_text}")
            elif confidence >= 0.50:
                st.warning(f"**{confidence_label}:** {status_text}")
            else:
                st.error(f"**{confidence_label}:** {status_text}")

            st.caption(
                "Confidence means the model's relative score among its four labels. "
                "It does not measure clinical certainty."
            )

        st.markdown('<p class="section-label">Step 3 · Probability breakdown</p>', unsafe_allow_html=True)

        probability_df = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        chart_col, table_col = st.columns([1.25, 1], gap="large")

        with chart_col:
            st.subheader("Class probability chart")
            st.bar_chart(
                probability_df.set_index("Class"),
                color="#176B87",
                use_container_width=True,
            )

        with table_col:
            st.subheader("Detailed probabilities")

            display_df = probability_df.copy()
            display_df["Probability"] = display_df["Probability"].map(lambda value: f"{value:.2%}")

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Important interpretation notes"):
            st.markdown(
                """
                - The model output describes which training label the uploaded image most resembles.
                - A high probability does not confirm that a medical condition is present.
                - Image quality, preprocessing, data imbalance, and differences from the training dataset can affect predictions.
                - Only an appropriately qualified clinician can interpret medical imaging in a clinical context.
                """
            )

    except Exception as error:
        st.error("The uploaded file could not be processed as an image.")
        st.code(str(error))


st.divider()
