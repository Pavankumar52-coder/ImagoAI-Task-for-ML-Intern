import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler (if needed, save it similarly using pickle)
scaler = StandardScaler()  # Assuming you have fitted it earlier
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI code
st.title('Corn Mycotoxin Prediction')
st.write("This app predicts DON concentration in corn samples using hyperspectral imaging data.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df.drop('DON_concentration', axis=1)
    y = df['DON_concentration']
    X_normalized = scaler.transform(X)
    y_pred = model.predict(X_normalized)

    results_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    st.write(results_df)

    st.write("Scatter plot of Actual vs. Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    st.pyplot(fig)