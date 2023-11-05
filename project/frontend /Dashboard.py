import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import googlemaps
from environs import Env
import gmaps
import pandas as pd
import numpy as np
import folium
import webbrowser
import os
import shutil

import folium
from folium.plugins import HeatMap


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
    with open(os.path.join("eogProject/project/frontend", uploaded_file.name), "wb") as f:
      f.write(uploaded_file.read())

    return uploaded_file
  else:
    st.error("Please upload a CSV file.")



# Get the uploaded file
prompt_file_upload()



location_leak, spread_leak, anomaly_detection = st.tabs(
    ["Location Leak", "Spread Leak", "Anomaly Detection"]
)


with location_leak:
  chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
  st.line_chart(chart_data)


with spread_leak:
 
 
 html_template = """
    <iframe src="{path_to_html}" width="100%" height="600"></iframe>
  """
 path_to_html = "/Users/prathmeshbhatt/Desktop/eogProject/project/frontend /satelliteMap.html" 
 
 with open(path_to_html,'r') as f: 
     html_data = f.read()

## Show in webpage
     st.header("Show an external HTML")
     st.components.v1.html(html_data,height=900)

 
 
# Display the map.
pass


st.title("CSV-Based Chatbot")
