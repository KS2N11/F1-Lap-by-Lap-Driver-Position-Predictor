import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Title with Emoji and Subheader
st.title('🏎️ F1 Lap-by-Lap Driver Position Predictor')
st.subheader("🚥 **Predict the next position and lap time of an F1 driver in real-time.**")

# Load Data
drivers = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/drivers.csv')
constructors = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/constructors.csv')
lap_times = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/lap_times.csv')
races = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/races.csv')
circuits = pd.read_csv('https://raw.githubusercontent.com/KS2N11/F1-Lap-by-Lap-Driver-Position-Predictor/refs/heads/master/datasets/circuits.csv')

# Sidebar Input with Styling
with st.sidebar:
    st.header('🏁 **Input Features**')
    circuit = st.selectbox('🏟️ Circuit Name', circuits['name'].tolist())
    year = st.selectbox('📅 Year', tuple(range(2024, 1949, -1)))
    driver = st.selectbox('👤 Driver Name', drivers['driverRef'].tolist())
    constructor = st.selectbox('🚗 Constructor Name', constructors['constructorRef'].tolist())
    grid_pos = st.slider('🔢 Grid Position', 1, 20, 10)
    final_pos = st.slider('🏅 Classification Position', 1, 20, 10)
    current_lap = st.slider('🔄 Current Lap', 1, 60, 30)

# Fetch IDs based on selected inputs
circuit_id = circuits.loc[circuits['name'] == circuit, 'circuitId'].iloc[0]
driver_id = drivers.loc[drivers['driverRef'] == driver, 'driverId'].iloc[0]
constructor_id = constructors.loc[constructors['constructorRef'] == constructor, 'constructorId'].iloc[0]
race_id = races.loc[races['circuitId'] == circuit_id, 'raceId'].iloc[0]

# DataFrame for input data
input_data = pd.DataFrame({
    'circuitId': [circuit_id], 'raceId': [race_id], 'year': [year], 
    'driverId': [driver_id], 'constructorId': [constructor_id],
    'driver_start_pos': [grid_pos], 'driver_fin_pos': [final_pos], 
    'curr_lap': [current_lap]
})

# Decision Tree Model (already trained)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)  # Assuming you already split and trained earlier

# Predict Driver Position
predicted_position = regressor.predict(input_data)[0]

# Progress Bar for Lap Completion
progress = current_lap / 60
st.progress(progress)

# Display Output with Metrics
st.markdown("## 🏁 **Race Prediction Summary**")
col1, col2 = st.columns(2)

with col1:
    st.metric("Current Lap", f"{current_lap}/60", delta=f"{60 - current_lap} remaining")
    st.metric("Grid Position", grid_pos)
    st.metric("Predicted Position", int(predicted_position))

with col2:
    st.markdown(f"**Circuit**: {circuit}")
    st.markdown(f"**Driver**: {driver} ({constructor})")
    st.markdown(f"**Year**: {year}")

# Lap Time Prediction using Second Model
input_data2 = pd.DataFrame({
    'raceId': [race_id], 'driverId': [driver_id], 
    'curr_lap': [current_lap], 'driver_curr_pos': [predicted_position]
})

predicted_lap_time_ms = regressor2.predict(input_data2)[0]

# Convert milliseconds to mm:ss.SSS format
minutes = int(predicted_lap_time_ms // 60000)
seconds = int((predicted_lap_time_ms % 60000) // 1000)
milliseconds = int(predicted_lap_time_ms % 1000)

# Display Predicted Lap Time in a Column Layout
st.markdown("### ⏱️ **Predicted Lap Time**")
st.write(f"**{minutes}** minutes, **{seconds}** seconds, **{milliseconds}** milliseconds")

# Highlight Final Prediction with Emoji
st.success(f"📊 **On Lap {current_lap}, {driver} is predicted to be in position P{int(predicted_position)}.**")
