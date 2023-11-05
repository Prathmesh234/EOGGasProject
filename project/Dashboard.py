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
#from datetime import time
from datetime import datetime, timedelta
from EOG_script import main_executor
import folium
from folium.plugins import HeatMap
import time

st.title('Gas Guard')

# Set the theme using the st.config object
st.sidebar.button("Welcome to Gas Guard!!")

st.markdown("**Please complete the following steps -**")

# Get the uploaded file

st.markdown("* First upload the weather_data csv file with the filename as 'weather_data.csv'")

def prompt_file_upload_weather():
    """Upload the weather data csv file"""
    uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"], key="weather")
    
    if uploaded_file is not None:
        with open(os.path.join("/Users/prathmeshbhatt/Desktop/eogProject/project/", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
           

        return uploaded_file
    else:
        st.error("Please upload a CSV file.")
prompt_file_upload_weather()

st.markdown("* Upload the sensor_readings csv file with the filename as 'sensor_readings.csv'")

def prompt_file_upload_sensor():
    """Upload the sensor data csv file"""
    uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"], key="sensor")
    
    if uploaded_file is not None:
        with open(os.path.join("/Users/prathmeshbhatt/Desktop/eogProject/project/", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
        main_executor()

        return uploaded_file
    else:
        st.error("Please upload a CSV file.")

prompt_file_upload_sensor()

st.markdown("* Please wait for the files to compile")

with st.spinner("Please wait while the sensor data is being prepared..."):
    while not os.path.exists("/Users/prathmeshbhatt/Desktop/eogProject/project/sensor_readings.csv"):
        time.sleep(1)
st.success("Data found!, please be patient...")


st.markdown("* **Download the results.csv file.** ")



df = pd.read_csv('results.csv')
# Create a download button
# Convert the DataFrame to a CSV string
data_string = df.to_csv(index=False)

# Encode the CSV string to UTF-8
data_bytes = data_string.encode('utf-8')
st.download_button(label="Download CSV", data=data_bytes, file_name="results.csv")
with st.spinner("Please wait while the result.csv file is being prepared..."):
    while not os.path.exists("/Users/prathmeshbhatt/Desktop/eogProject/project/results.csv"):
        time.sleep(5)
st.success("Results.csv is ready!!")

sensor_locations, location_leak, spread_leak, anomaly_detection = st.tabs(
    ["Sensor Locations", "Leak Locations", "Spread Leak", "Anomaly Detection"]
)

with location_leak:
  
  t = st.time_input('Input time of the day in hh:mm format example 00:30', value=None)
  if t is not None:
    st.write('Looking for ', t)
    appointment = t  # 2:30 PM
    hours = appointment.hour
    minutes = appointment.minute
    # Extract hours and minutes from the time string
    

    # Convert HH:MM to seconds
    total_seconds = (hours * 3600) + (minutes * 60)

    print(f'Total seconds: {total_seconds}')

    results = pd.read_csv("results.csv")

    location = results.iloc[total_seconds]["location"]

    leak_lat_long = {"4W": [40.595945, -105.140321],"4S": [40.595641, -105.140301], "5W": [40.595660, -105.139428], "5S": [40.595926, -105.139408], "4T": [40.595792, -105.139862]}

    latitudes = []  # Add your latitude values here
    longitudes = []  # Add your longitude values here
    labels = []
    count = 0
    for id in location.split('|'):
        if id == "None":
            # Display no leaks  
            break
        latitudes += [leak_lat_long[id][0]]
        longitudes += [leak_lat_long[id][1]]
        labels += [id]

    # Create a map centered at a specific location (optional)
    map_center = [40.595792, -105.139862]  # Center the map around the average of latitudes and longitudes
    map_obj = folium.Map(location=map_center, zoom_start=100)  # You can adjust the zoom level as needed

    # Add markers for each latitude and longitude pair
    for lat, lon, label in zip(latitudes, longitudes, labels):
        folium.Marker(location=[lat, lon], popup=folium.Popup(label, parse_html=True)).add_to(map_obj)

    # Save the map as an HTML file
    map_obj.save('map.html')
    path_to_html2 = "/Users/prathmeshbhatt/Desktop/eogProject/project/map.html"
    html_template = """
        <iframe src="{path_to_html2}" width="100%" height="600"></iframe>
    """
    with open(path_to_html2,'r') as f: 
        html_data = f.read()

    ## Show in webpage
        st.header("Map spread of the leak")
        st.components.v1.html(html_data,height=900)


with spread_leak:
 
 
 html_template = """
    <iframe src="{path_to_html}" width="100%" height="600"></iframe>
  """
 path_to_html = "/Users/prathmeshbhatt/Desktop/eogProject/project/satelliteMap.html" 
 
 with open(path_to_html,'r') as f: 
     html_data = f.read()

## Show in webpage
     st.header("Show an external HTML")
     st.components.v1.html(html_data,height=900)

 
 
# Display the map.
pass


