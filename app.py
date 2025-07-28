import streamlit as st
import pickle
import numpy as np

# === Load model and metadata ===
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("iris_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

feature_names = metadata["feature_names"]
target_names = metadata["target_names"]

# === Streamlit UI ===
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier")
st.markdown("Adjust the flower measurements to predict the species.")

# === Create sliders dynamically ===
input_values = []
feature_ranges = {
    "sepal length (cm)": (4.0, 8.0),
    "sepal width (cm)": (2.0, 4.5),
    "petal length (cm)": (1.0, 7.0),
    "petal width (cm)": (0.1, 2.5)
}

for name in feature_names:
    min_val, max_val = feature_ranges[name]
    default_val = round((min_val + max_val) / 2, 2)
    val = st.slider(name.title(), min_val, max_val, default_val)
    input_values.append(val)

# === Predict when user clicks button ===
if st.button("🔍 Predict"):
    input_array = np.array([input_values])
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]

    # Display prediction
    st.success(f"🌼 Predicted Species: **{target_names[prediction]}**")

    # Display probabilities
    st.subheader("Prediction Confidence:")
    for i, species in enumerate(target_names):
        st.write(f"- {species}: **{probabilities[i]*100:.2f}%**")

    # Optionally show the raw input values
    with st.expander("🔬 See model input values"):
        input_dict = dict(zip(feature_names, input_values))
        st.json(input_dict)

