import streamlit as st
import pandas as pd

st.title('🏎️ F1-Lap-by-Lap-Driver-Position-Predictor')

drivers = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/drivers.csv')
driver_standings = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/driver_standings.csv')
constructors = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/constructors.csv')
constructor_standings = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/constructor_standings.csv')
races = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/races.csv')
circuits = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/circuits.csv')
lap_times = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/lap_times.csv')
results = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/results.csv')

with st.expander('Raw Data'):
  drivers
