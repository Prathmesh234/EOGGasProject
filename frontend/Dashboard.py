import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import googlemaps



APIKey = "AIzaSyAh49B0BiaXa_F61OL9aY8rrK1Vq8F89DM"



st.title('Gas Leak Detector')


# Set the theme using the st.config object
st.sidebar.button("Leak number 1")
st.sidebar.button("Leak number 2")
st.sidebar.button("Leak number 3")
st.sidebar.button("Leak number 4")
st.sidebar.button("Leak number 5")



def prompt_file_upload():
  """Upload the sensor data csv file"""
  uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])
  if uploaded_file is not None:
    return uploaded_file
  else:
    st.error("Please upload a CSV file.")


# Get the uploaded file
uploaded_file = prompt_file_upload()

# If the user uploaded a file, process it here
if uploaded_file is not None:
  # Read the CSV file into a Pandas DataFrame
  df = pd.read_csv(uploaded_file)

  # Display the contents of the DataFrame in a Streamlit table
  st.table(df)


location_leak, spread_leak, anomaly_detection = st.tabs(
    ["Location Leak", "Spread Leak", "Anomaly Detection"]
)


with location_leak:
  chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
  st.line_chart(chart_data)


with spread_leak:
 

  pass