import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

st.title('🏎️ F1-Lap-by-Lap-Driver-Position-Predictor')
image = st.image(link = 'data:image/jpeg;base64,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')

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
  d1 = drivers.head(1000)
  d1
  st.write('**Driver Standings**')
  d2 = driver_standings.head(1000)
  d2
  st.write('**Constructor Info**')
  d3 = constructors.head(1000)
  d3
  st.write('**Constructor Standings**')
  d4 = constructor_standings.head(1000)
  d4
  st.write('**Race Info**')
  d5 = races.head(1000)
  d5
  st.write('**Circuit Info**')
  d6 = circuits.head(1000)
  d6
  st.write('**Lap Times**')
  d7 = lap_times.head(1000)
  d7
  st.write('**Results**')
  d8 = results.head(1000)
  d8


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
  d9 = dataset.head(1000)
  d9


#Model 1
X = dataset.drop(columns = ['driver_curr_pos'], axis = 0)
y = dataset['driver_curr_pos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

r1 = r2_score(y_test, y_pred)

#Model 2
X2 = lap_times.drop(columns = ['milliseconds', 'curr_lap_time'], axis = 0)
y2 = lap_times['milliseconds']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

regressor2 = DecisionTreeRegressor(random_state = 0)
regressor2.fit(X2_train, y2_train)

y2_pred = regressor2.predict(X2_test)
r2 = r2_score(y2_test, y2_pred)



circuit_names = circuits['name']
circuit_names = circuit_names.values.tolist()
circuit_names = tuple(circuit_names)

driver_names = drivers['driverRef']
driver_names = driver_names.values.tolist()
driver_names = tuple(driver_names)

constructor_names = constructors['constructorRef']
constructor_names = constructor_names.values.tolist()
constructor_names = tuple(constructor_names)

years = tuple(range(2024, 1949, -1))

with st.sidebar:
  st.header('**Input Features**')
  circuit = st.selectbox('Circuit Name', circuit_names)
  year = st.selectbox('Year', years)
  driver = st.selectbox('Driver Name', driver_names)
  constructor = st.selectbox('Constructor Name', constructor_names)
  grid_pos = st.slider('Grid Position', 1, 20, 10)
  final_pos = st.slider('Classification Position', 1, 20, 10)
  current_lap = st.slider('Current Lap', 1, 60, 30)

  #Creating a Dataframe
  circuit_id = circuits.loc[circuits['name'] == circuit, 'circuitId'].iloc[0]
  driver_id = drivers.loc[drivers['driverRef'] == driver, 'driverId'].iloc[0]
  constructor_id = constructors.loc[constructors['constructorRef'] == constructor, 'constructorId'].iloc[0]
  race_id = races.loc[(races['circuitId']==circuit_id), 'raceId'].iloc[0]
  round = races.loc[(races['circuitId'] == circuit_id), 'round'].iloc[0]

  input_data1 = pd.DataFrame(
    {
        'circuitId': [circuit_id],
        'raceId': [race_id],
        'year': [year],        
        'round': [round],
        'driverId': [driver_id],
        'driver_fin_pos': [final_pos],        
        'constructorId': [constructor_id],
        'curr_lap': [current_lap],
        'driver_start_pos': [grid_pos]
    }
  )

if(current_lap == 60):
  output = final_pos
else:
  output = regressor.predict(input_data1)
  output = output[0]

input_data2 = pd.DataFrame(
    {
        'raceId': [race_id],
        'driverId': [driver_id],
        'curr_lap': [current_lap],
        'driver_curr_pos': [output]
    }
  )

output2 = regressor2.predict(input_data2)
output2 = output2[0]

output_in_min = output2 / 60000
minutes = output2 // 60000
seconds = (output2 % 60000) // 1000
milliseconds = output2 % 1000

output = int(output)
output = str(output)

st.write('**Position of**', driver, '**from**', constructor, '**on lap**', current_lap, '**at**', circuit, '**is P**',output)
st.write("Lap Time- ", minutes,":",seconds,".", milliseconds)
