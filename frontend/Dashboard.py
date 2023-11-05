import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import googlemaps



APIKey = "AIzaSyAh49B0BiaXa_F61OL9aY8rrK1Vq8F89DM"
map_client = googlemaps.Client(APIKey)



st.title('Gas Leak Detector')


# Set the theme using the st.config object
st.sidebar.button("Leak number 1")
st.sidebar.button("Leak number 2")
st.sidebar.button("Leak number 3")
st.sidebar.button("Leak number 4")
st.sidebar.button("Leak number 5")
st.sidebar.button("Leak number 6")




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

# tabs for the following sets of information
sensor_location, location_leak, spread_leak, anomaly_detection = st.tabs(
    ["Sensor Locations","Location Leak", "Spread Leak", "Anomaly Detection"]
)

with sensor_location:
  ####import results.csv
  #data = pd.read_csv("results.csv", delimiter= ',')
  import streamlit as st
  from bokeh.plotting import figure

  x = [1, 2, 3, 4, 5]
  y = [6, 7, 2, 4, 5]

  p = figure(
      title='simple line example',
      x_axis_label='x',
      y_axis_label='y')

  p.line(x, y, legend_label='Trend', line_width=2)

  st.bokeh_chart(p, use_container_width=True)
  pass

with location_leak:
  chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
  st.line_chart(chart_data)


with spread_leak:
 

  pass

