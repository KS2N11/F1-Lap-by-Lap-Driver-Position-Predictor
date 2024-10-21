import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

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
  st.write('**Driver Info**')
  drivers
  st.write('**Driver Standings**')
  driver_standings
  st.write('**Constructor Info**')
  constructors
  st.write('**Constructor Standings**')
  constructor_standings
  st.write('**Race Info**')
  races
  st.write('**Circuit Info**')
  circuits
  st.write('**Lap Times**')
  lap_times
  st.write('**Results**')
  results


circuits = circuits.drop(columns = ['lat', 'lng', 'alt', 'url'], axis = 0)
races = races.drop(columns = ['date', 'time', 'url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time'], axis = 0)

merge1 = circuits.merge(races, on = 'circuitId', how = 'right')
merge1 = merge1.drop(columns = ['circuitRef', 'name_x', 'location', 'country', 'name_y'], axis = 0)

drivers = drivers.drop(columns = ['number', 'dob', 'url'], axis = 0)
driver_standings = driver_standings.rename(columns = {'position': 'driver_fin_pos'})
driver_standings = driver_standings.drop(columns = ['positionText', 'wins'], axis = 0)

merge2 = drivers.merge(driver_standings, on = 'driverId', how = 'right')
merge2 = merge2.drop(columns = ['driverRef', 'code', 'forename', 'surname', 'nationality', 'driverStandingsId', 'points'], axis = 0)

constructors = constructors.drop(columns = ['url', 'nationality'], axis = 0)
constructor_standings = constructor_standings.rename(columns = {'position': 'constructor_fin_pos'})
constructor_standings = constructor_standings.drop(columns = ['positionText', 'wins'], axis = 0)

merge3 = constructors.merge(constructor_standings, on = 'constructorId', how = 'right')
merge3 = merge3.drop(columns = ['constructorRef', 'name', 'constructorStandingsId', 'points'], axis = 0)

merge1_2 = merge1.merge(merge2, on = 'raceId', how = 'left')
merge1_2_3 = merge1_2.merge(merge3, on = 'raceId', how = 'left')

lap_times = lap_times.rename(columns = {'position': 'driver_curr_pos', 'lap': 'curr_lap', 'time': 'curr_lap_time'})

merge4 = merge1_2_3.merge(lap_times, on = ['raceId', 'driverId'], how = 'left')

results = results.rename(columns = {'position': 'driver_fin_pos', 'grid': 'driver_start_pos', 'time': 'fin_time', 'rank': 'driver_fastest_lap_rank', 'laps': 'total_laps'})
results = results.drop(columns = ['number', 'positionText', 'positionOrder', 'points', 'milliseconds', 'fin_time', 'fastestLap', 'driver_fastest_lap_rank', 'fastestLapTime', 'fastestLapSpeed', 'statusId'], axis = 0)

merge4['driver_fin_pos'] = pd.to_numeric(merge4['driver_fin_pos'], errors='coerce') # Convert to numeric, handling errors
results['driver_fin_pos'] = pd.to_numeric(results['driver_fin_pos'], errors='coerce') # Convert to numeric, handling errors

finalmerge = merge4.merge(results, on=['raceId', 'driverId', 'constructorId', 'driver_fin_pos'], how='right')
finalmerge = finalmerge.drop(columns = ['curr_lap_time', 'resultId', 'constructor_fin_pos', 'total_laps', 'milliseconds'])

dataset = finalmerge
dataset.dropna(subset=['driver_curr_pos'], inplace=True)

with st.expander('Prepared Data'):
  st.write('**Final Dataset**')
  dataset

X = dataset.drop(columns = ['driver_curr_pos'], axis = 0)
y = dataset['driver_curr_pos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

r = r2_score(y_test, y_pred)

driver_names = drivers['driverRef']
x = driver_names.values.tolist()
t = tuple(x)
t

with st.sidebar:
  st.header('**Input Features**')
  





