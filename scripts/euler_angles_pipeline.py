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
parser.add_argument("--cutoff_hz", help="Fréquence de coupure (Hz)")
parser.add_argument("--freq_measure", type=int, help="Fréquence d'échantillonage' (Hz)")
parser.add_argument("--time_windows", type=int, help="Taille fenêtre' (s)")
parser.add_argument("--euler_angles", help="Euler angles yes or no")
parser.add_argument("--features_selection", help="Nb of best features")
parser.add_argument("--behavior1", help="First behavior to predict")
parser.add_argument("--behavior2", help="Second behavior to predict", nargs='?', default=None)
parser.add_argument("--behavior3", help="Third behavior to predict", nargs='?', default=None)
parser.add_argument("--behavior4", help="Fourth behavior to predict", nargs='?', default=None)

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
current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)

behavior1 = args.behavior1
behavior2 = args.behavior2
behavior3 = args.behavior3
behavior4 = args.behavior4

behaviours = [behavior1,behavior2,behavior3,behavior4]
behaviours_names = '_'.join(behaviours)

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


path = os.path.join(home_directory, id_expe, str(time_windows)+'s_window')

            
timeseries=[]
path_file=[]

if reference_vector_computation=='' :
    reference_vector_computation2=reference_vector_computation
if reference_vector_computation!='' : 
    reference_vector_computation2="_"+reference_vector_computation
path_file=path+'/'+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+behaviours_names+"_"+str(time_windows)+"s"+".csv"
path_timeseries=path+'/'+'timeseries_'+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+behaviours_names+"_"+str(time_windows)+"s"+".csv"

print('Euler angles script')
print('Path file =', path_file)
print('Path timeseries =', path_timeseries)

filepath = Path(path+'/'+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+behaviours_names+'_'+str(time_windows)+'s'+euler_angles+'.csv')
filepath2 = Path(path+'/'+'timeseries_'+id_expe+reference_vector_computation2+'_'+filter+'_'+cutoff_hz_str+behaviours_names+'_'+str(time_windows)+'s'+euler_angles+'.csv')

if euler_angles=='_euler_angles' :

    # Check if the output files already exist
    if filepath.exists() and filepath2.exists():
        print(f"The output files {filepath} and {filepath2} already exist. Skipping euler_angles calculation.")
        file_exists = True

    
    timeseries = pd.read_csv(path_timeseries,sep=",")
    timeseries['TIME'] = pd.to_datetime(timeseries['TIME'])
    timeseries = timeseries.drop(['Unnamed: 0'], axis=1)
    timeseries = timeseries.drop(['id_goat'], axis=1)

    timeseries['roll'] = np.arctan2(timeseries['ACCyf'], timeseries['ACCzf'])
    timeseries['pitch'] = np.arctan2(-timeseries['ACCxf'], np.sqrt(timeseries['ACCyf']**2 + timeseries['ACCzf']**2))
    
    filepath2.parent.mkdir(parents=True, exist_ok=True)  
    timeseries.to_csv(filepath2)
    
    try:
        df = pd.read_csv(path_file, sep=",")
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath) 
    except Exception as e:
        file_exists = True



if euler_angles=='' :
    
    # Check if the output files already exist
    if filepath.exists() and filepath2.exists():
        print(f"The output files {filepath} and {filepath2} already exist. Skipping euler_angles calculation.")
        file_exists = True

    timeseries = pd.read_csv(path_timeseries,sep=",")
    filepath2.parent.mkdir(parents=True, exist_ok=True)  
    timeseries.to_csv(filepath2)
    
    try:
        df= pd.read_csv(path_file,sep=",")
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath) 
    except Exception as e:
        file_exists = True

