import numpy as np
import pandas as pd
import scipy as signal
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
import plotly.graph_objects as go
import glob
import os.path
import datetime
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from math import*
import tsfresh
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters,EfficientFCParameters,ComprehensiveFCParameters
import re
import argparse


# Créez un objet ArgumentParser
parser = argparse.ArgumentParser()

# Ajoutez les arguments que vous souhaitez prendre en charge
parser.add_argument("--id_expe", help="ID de l'expérience")
parser.add_argument("--filter", help="Filtrer les données ou non (yes/no)")
parser.add_argument("--reference_vector_computation", help="Type de calcul du vecteur de référence")
parser.add_argument("--cutoff_hz",help="Fréquence de coupure (Hz)")
parser.add_argument("--freq_measure", type=int, help="Fréquence d'échantillonage' (Hz)")
parser.add_argument("--time_windows", type=int, help="Taille fenêtre' (s)")
parser.add_argument("--euler_angles", help="Euler angles yes or no")
parser.add_argument("--features_selection", help="Nb of best features")
parser.add_argument("--behavior1", help="First behavior to predict")
parser.add_argument("--behavior2", help="Second behavior to predict", nargs='?', default=None)
parser.add_argument("--behavior3", help="Third behavior to predict", nargs='?', default=None)
parser.add_argument("--behavior4", help="Fourth behavior to predict", nargs='?', default=None)
parser.add_argument("--processes", type=int, help="Processes parallelization")

# Analysez les arguments de ligne de commande
args = parser.parse_args()

# Accédez aux arguments avec les attributs correspondants
id_expe = args.id_expe
euler_angles = args.euler_angles 
filter = args.filter
time_windows = args.time_windows
reference_vector_computation = args.reference_vector_computation if args.reference_vector_computation else ""
cutoff_hz = args.cutoff_hz
freq_measure = args.freq_measure
features_selection = args.features_selection
behavior1 = args.behavior1
behavior2 = args.behavior2
behavior3 = args.behavior3
behavior4 = args.behavior4
processes = args.processes

features_parameters = EfficientFCParameters() #MinimalFCParameters(), EfficientFCParameters(), ComprehensiveFCParameters() or None
behaviours = [behavior1,behavior2,behavior3,behavior4]
behaviours_names = '_'.join(sorted(behaviours))

current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)
path=home_directory+'/'+id_expe+'/'+str(time_windows)+'s_window/'


if euler_angles=='yes' :
    euler_angles='_euler_angles'
if euler_angles=='no' :
    euler_angles=''

if filter=='yes' :
    filter='filter'
    cutoff_hz_str = (re.sub(r'[^\w\s]', '', str(cutoff_hz)))+"hz_"
if filter=='no' :
    filter='no_filter'
    cutoff_hz_str = ""

timeseries=[]
path_file=[]

if reference_vector_computation=='' :
    reference_vector_computation2=reference_vector_computation
if reference_vector_computation!='' : 
    reference_vector_computation2="_"+reference_vector_computation
path_file=path+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+behaviours_names+"_"+str(time_windows)+"s"+euler_angles+".csv"
path_timeseries=path+'timeseries_'+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+behaviours_names+"_"+str(time_windows)+"s"+euler_angles+".csv"

print('Features engineering script')
print('Path file =', path_file)
print('Path timeseries =', path_timeseries)


timeseries = pd.read_csv(path_timeseries,sep=",")
timeseries['TIME'] = pd.to_datetime(timeseries['TIME'])
if 'Unnamed: 0' in timeseries.columns:
    timeseries = timeseries.drop(['Unnamed: 0'], axis=1)
if 'Unnamed: 0.1' in timeseries.columns:
    timeseries = timeseries.drop(['Unnamed: 0.1'], axis=1)
if 'id_goat' in timeseries.columns:
    timeseries = timeseries.drop(['id_goat'], axis=1)
if 'start_window' in timeseries.columns:
    timeseries = timeseries.drop(['start_window'], axis=1)

path_preprocessed_data=home_directory+'/preprocessed_data'
filepath = Path(home_directory+'/preprocessed_data/'+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+str(time_windows)+'s_'+behaviours_names+'_'+type(features_parameters).__name__+euler_angles+'.csv')
if not os.path.exists(path_preprocessed_data):
    os.makedirs(path_preprocessed_data, exist_ok=True)
    
# Check if the output files already exist
if filepath.exists() :
    print(f"The output files {filepath} already exist. Skipping features engineering.")
    exit()

split_index = len(timeseries) // 2
timeseries_part1 = timeseries.iloc[:split_index]
timeseries_part2 = timeseries.iloc[split_index:]

for col in timeseries_part1.columns:
    if col=='TIME' :
        continue
    if col=='start_window_id':
        continue

    # Extract features for each column separately
    extracted_features1 = extract_features(timeseries_part1[['start_window_id','TIME',col]], column_id="start_window_id", column_sort="TIME", n_jobs=0, default_fc_parameters=features_parameters)
    path_running=home_directory+'/scripts/running'
    if not os.path.exists(path_running):
        os.makedirs(path_running, exist_ok=True)
        os.makedirs(path_running+'/timeseries1', exist_ok=True)
    extracted_features1.to_csv(f"{home_directory}/scripts/running/timeseries1/extracted_features_part1_{col}.csv")




extracted_features_concatenated1 = None
for col in timeseries_part1.columns:
    if col=='TIME' :
        continue
    if col=='start_window_id':
        continue
    if col=='Unnamed: 0':
        continue 
    # Load the extracted features for this column from disk
    extracted_features1 = pd.read_csv(home_directory+'/scripts/running/timeseries1/extracted_features_part1_'+col+'.csv')
    extracted_features1 = extracted_features1.rename(columns={'Unnamed: 0': 'start_window_id'})
    # Concatenate with other columns
    if extracted_features_concatenated1 is None:
        extracted_features_concatenated1 = extracted_features1
    else:
        extracted_features_concatenated1 = pd.merge(extracted_features_concatenated1, extracted_features1, on='start_window_id')

for col in timeseries_part2.columns:
    if col=='TIME' :
        continue
    if col=='start_window_id':
        continue
    
    # Extract features for each column separately
    extracted_features2 = extract_features(timeseries_part2[['start_window_id','TIME',col]], column_id="start_window_id", column_sort="TIME", n_jobs=0, default_fc_parameters=features_parameters)
    path_timeseries2=home_directory+'/scripts/running/timeseries2'
    if not os.path.exists(path_timeseries2):
        os.makedirs(path_timeseries2, exist_ok=True)
    extracted_features2.to_csv(f"{home_directory}/scripts/running/timeseries2/extracted_features_part2_{col}.csv")

extracted_features_concatenated2 = None
for col in timeseries_part2.columns:
    if col == 'TIME':
        continue
    if col=='start_window_id':
        continue
    if col=='Unnamed: 0':
        continue 

    # Load the extracted features for this column from disk
    extracted_features2 = pd.read_csv(home_directory+'/scripts/running/timeseries2/extracted_features_part2_'+col+'.csv')
    extracted_features2 = extracted_features2.rename(columns={'Unnamed: 0': 'start_window_id'})

    # Concatenate with other columns
    if extracted_features_concatenated2 is None:
        extracted_features_concatenated2 = extracted_features2
    else:
        extracted_features_concatenated2 = pd.merge(extracted_features_concatenated2, extracted_features2, on='start_window_id')

# Merge if df_window is successfully loaded
merge_extracted_features = pd.concat([extracted_features_concatenated1, extracted_features_concatenated2])

try:
    df_window = pd.read_csv(path_file, sep=",")

    # Drop 'Unnamed' columns if they exist
    if 'Unnamed: 0' in df_window.columns:
        df_window = df_window.drop(['Unnamed: 0'], axis=1)
    if 'Unnamed: 0.1' in df_window.columns:
        df_window = df_window.drop(['Unnamed: 0.1'], axis=1)

    # Convert "yes" and "no" to 1 and 0
    df_window = df_window.replace("yes", 1)
    df_window = df_window.replace("no", 0)

    merge_extracted_features = merge_extracted_features.merge(df_window, on='start_window_id', how='inner')


except FileNotFoundError:
    print("Features engineering done without labeled data")
except pd.errors.ParserError:
    print("Features engineering done without labeled data")


filepath.parent.mkdir(parents=True, exist_ok=True)  
merge_extracted_features.to_csv(filepath)
 
filepath1=home_directory+"/scripts/running/timeseries1"
files = os.listdir(filepath1)

# Iterate over each file and delete it
for file in files:
    file_path = os.path.join(filepath1, file)
    os.remove(file_path)

filepath1=home_directory+"/scripts/running/timeseries2"
files = os.listdir(filepath1)

# Iterate over each file and delete it
for file in files:
    file_path = os.path.join(filepath1, file)
    os.remove(file_path)
