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
import re
import argparse
from multiprocessing import Pool


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
parser.add_argument("--processes", type=int, help="Processes parallelization")


# Analysez les arguments de ligne de commande
args = parser.parse_args()

# Accédez aux arguments avec les attributs correspondants
id_expe = args.id_expe
euler_angles = args.euler_angles 
filter = args.filter
time_windows = args.time_windows
reference_vector_computation = args.reference_vector_computation if args.reference_vector_computation else ""
cutoff_hz = args.cutoff_hz if args.cutoff_hz else ""
freq_measure = args.freq_measure
features_selection = args.features_selection
processes=args.processes

behavior1 = args.behavior1
behavior2 = args.behavior2
behavior3 = args.behavior3
behavior4 = args.behavior4

behaviours = [behavior1,behavior2,behavior3,behavior4]
behaviours_names = '_'.join(behaviours)

cutoff_hz = re.sub(r'[^\w\s]', '', str(cutoff_hz))

current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)

path = os.path.join(home_directory, id_expe)
all_files=[]

if reference_vector_computation=='' :
    reference_vector_computation2=reference_vector_computation
if reference_vector_computation!='' : 
    reference_vector_computation2="_"+reference_vector_computation

if filter=='yes' :
    filter='filter'
if filter=='no' :
    filter='no_filter'

if filter=="filter" :
    all_files = glob.glob(path  +'/filtered/'+ id_expe + "_[0-9][0-9][0-9][0-9][0-9]" + reference_vector_computation2 +'_' +filter + "_" + cutoff_hz + "hz_" + behaviours_names + ".csv")
if filter=="no_filter" :
    all_files = glob.glob(path  +'/filtered/'+ id_expe + "_[0-9][0-9][0-9][0-9][0-9]" + reference_vector_computation2 +'_' +filter + '_' + behaviours_names + ".csv")

print('Windows tache de prediction script')
print('Nombre de fichiers sélectionnés =', len(all_files) )
print('Files =', all_files)

file_exists = False

cutoff_hz_str = re.sub(r'[^\w\s]', '', str(cutoff_hz))
cutoff_hz_str = cutoff_hz_str+'hz_'

if filter == 'no_filter' :
    cutoff_hz_str=''
if reference_vector_computation!='' :
    reference_vector_computation=reference_vector_computation+'_'


# Check if the file already exists
print(path +'/' + str(time_windows) + 's_window/'  + id_expe + '_' + reference_vector_computation + filter + '_' + cutoff_hz_str + behavior1 + '_' + behavior2 + '_' + behavior3 + '_' + behavior4  + '_' + str(time_windows) + 's.csv')
filtered_file_path = Path(home_directory + '/' + str(time_windows) + 's_window/'  + id_expe + '_' + reference_vector_computation + filter + '_' + cutoff_hz_str + behavior1 + '_' + behavior2 + '_' + behavior3 + '_' + behavior4  + '_' + str(time_windows) + 's.csv')
if filtered_file_path.exists():
    print(f"The filtered file {filtered_file_path} already exists. Skipping window creation.")
    file_exists = True


# Check if the timeseries file already exists
timeseries_file_path = Path(path + '/' + str(time_windows) + 's_window/timeseries_' + id_expe + '_' + reference_vector_computation + filter + '_' + cutoff_hz + behavior1 + '_' + behavior2 + '_' + behavior3 + '_' + behavior4 + '_' + str(time_windows) + 's.csv')
if timeseries_file_path.exists():
    print(f"The timeseries file {timeseries_file_path} already exists. Skipping window creation.")
    file_exists = True


# Check if the output folder already exists
path_folder=path+ '/' + str(time_windows) + 's_window'
if not os.path.exists(path_folder):
    os.makedirs(path_folder, exist_ok=True)

# If the file(s) already exist, no need to create the windows, so just exit the script
if file_exists:
    exit()

# Functions
def count_behavior(behavior, category, window, df, df_count):

    category_behavior = category
    window_count=0
    count=0
    while window_count < window.shape[0] : 
        if window.iloc[window_count][category_behavior]==behavior :
            count+=1
        window_count+=1
    proportion = (count*100)/window.shape[0]

    if count < (window.shape[0]/2) :
        df.loc[df_count,'yes_no_'+behavior] = 'no'

    else :
        df.loc[df_count,'yes_no_'+behavior] = 'yes'
    df.loc[df_count,'proportion_'+behavior] = proportion
            

def find_behavior_categories(raw_data, specified_behaviors, behavior_columns):
    behavior_categories = {}
    for behavior in specified_behaviors:
        # found variable is unused, removing it
        for column in behavior_columns:
            if behavior in raw_data[column].unique():
                behavior_categories[behavior] = column
                # found variable is unused, removing it
                break  
    return behavior_categories


df = pd.DataFrame()
df1 = pd.DataFrame()

# initialzation time of milking
filtered_data = pd.read_csv(all_files[0],sep=",")
filtered_data['TIME'] = pd.to_datetime(filtered_data['TIME'])

specified_behaviors = [args.behavior1, args.behavior2, args.behavior3, args.behavior4]
behavior_columns = filtered_data.columns.tolist()

if len(filtered_data.columns) >= 7 :
    behavior_categories = find_behavior_categories(filtered_data, specified_behaviors, behavior_columns)
    category1 = behavior_categories.get(args.behavior1)
    category2 = behavior_categories.get(args.behavior2)
    category3 = behavior_categories.get(args.behavior3)
    category4 = behavior_categories.get(args.behavior4)
    category_columns = behavior_categories.values()

start_milking_morning=filtered_data.iloc[filtered_data.shape[0]-1]['TIME'] + timedelta(days=2) 
end_milking_morning=filtered_data.iloc[0]['TIME'] - timedelta(days=2)
start_milking_after=filtered_data.iloc[filtered_data.shape[0]-1]['TIME'] + timedelta(days=2)
end_milking_after=filtered_data.iloc[0]['TIME'] - timedelta(days=2)

df_count=0
nb_lines=time_windows*freq_measure

def process_file(file):
    filtered_data = pd.read_csv(file, sep=",")
    filtered_data['TIME'] = pd.to_datetime(filtered_data['TIME'])
    for column in ['ACCxf', 'ACCyf', 'ACCzf']:
        filtered_data[column] = pd.to_numeric(filtered_data[column], errors='coerce')
    filtered_data = filtered_data.dropna()

    t = 0
    idchevre = str(filtered_data['idgoat'][0])
    filtered_data['norm'] = np.sqrt(filtered_data['ACCxf']**2 + filtered_data['ACCyf']**2 + filtered_data['ACCzf']**2)

    local_df = pd.DataFrame()
    local_df1 = pd.DataFrame()

    while t < filtered_data.shape[0]:
        window = filtered_data.iloc[t:(t + nb_lines), :]
        df1_window = pd.DataFrame()
        mycolumns = []
        columns_to_check = ['TIME', 'ACCxf', 'ACCyf', 'ACCzf', 'norm', 'ACCxf_r', 'ACCyf_r', 'ACCzf_r']

        for col in columns_to_check:
            try:
                col_loc = filtered_data.columns.get_loc(col)
                mycolumns.append(col_loc)
            except KeyError:
                pass

        df1_window = filtered_data.iloc[t:(t + nb_lines), mycolumns]

        if len(filtered_data.columns) >= 8 :
            count_behavior(behavior1, category1, window, local_df, df_count)
            count_behavior(behavior2, category2, window, local_df, df_count)
            count_behavior(behavior3, category3, window, local_df, df_count)
            count_behavior(behavior4, category4, window, local_df, df_count)

        local_df.loc[len(local_df), 'start_window'] = filtered_data.iloc[t]['TIME']
        local_df.loc[len(local_df) - 1, 'start_window_id'] = str(str(filtered_data.iloc[t]['TIME']) + '_' + str(idchevre)).replace(" ", "_")

        df1_window.loc[:, 'start_window'] = filtered_data.iloc[t]['TIME']
        df1_window.loc[:, 'start_window_id'] = str(str(filtered_data.iloc[t]['TIME']) + '_' + str(idchevre)).replace(" ", "_")
        df1_window.loc[:, 'id_goat'] = idchevre

        local_df1 = pd.concat([local_df1, df1_window]).reset_index(drop=True)
        t = t + nb_lines

    return local_df, local_df1

with Pool(processes=processes) as pool:
    results = pool.map(process_file, all_files)

# Concatenate results from all processes
df = pd.concat([result[0] for result in results]).reset_index(drop=True)
df1 = pd.concat([result[1] for result in results]).reset_index(drop=True)

# Write csv / excel
if len(filtered_data.columns) >= 8 :
    filepath = Path(path +'/'+str(time_windows)+'s_window/'+id_expe+'_'+reference_vector_computation+filter+'_'+cutoff_hz_str+behavior1+'_'+behavior2+'_'+behavior3+'_'+behavior4+'_'+str(time_windows)+'s.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath) 

filepath = Path(path +'/'+str(time_windows)+'s_window/'+'timeseries_'+id_expe+'_'+reference_vector_computation+filter+'_'+cutoff_hz_str+behavior1+'_'+behavior2+'_'+behavior3+'_'+behavior4+'_'+str(time_windows)+'s.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)  
df1.to_csv(filepath) 
