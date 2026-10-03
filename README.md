# Kidney CT Medical Imaging ML 

**Anualithe uwayo**  


---

## Contents

- `kidney_ct_notebook.ipynb`  
  Documented Jupyter notebook covering Task 0 and Part A:
  - Dataset description and scope
  - Data preparation and train/validation/test splits
  - MobileNetV2 model definition and training
  - Evaluation metrics (accuracy, macro F1, classification report, confusion matrix)
  - Model export and limitations/ethics discussion

- `app.py`  
  Streamlit web application that:
  - Loads the saved Keras model
  - Accepts PNG/JPG/JPEG image uploads
  - Displays the uploaded image
  - Shows predicted class and confidence
  - Shows class-probability table
  - Displays an uncertainty warning when confidence is low

- `kidney_ct_mobilenetv2_balanced_best.keras`  
  Saved TensorFlow/Keras model trained on the kidney CT dataset.

- `requirements.txt`  
  Python package versions required to run the environment and app.

---

## How to run locally

1. Create and activate a Python virtual environment (Python 3.10 recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. In the browser, open the URL shown (e.g. `http://localhost:8501`) and upload a PNG/JPG/JPEG CT-like image to see a prediction.

---

## Notes and limitations

- This is a **student coursework prototype**, not a medical device.
- The model has limited accuracy and should **not** be used for real diagnosis.
- Results are for **educational and demonstration purposes only**.
- Low-confidence predictions are explicitly flagged in the app.

---

## Troubleshooting

If there are issues running the app, check:

- Python version (3.10 recommended)
- That all packages in `requirements.txt` are installed
- That the `.keras` model file is in the same folder as `app.py`