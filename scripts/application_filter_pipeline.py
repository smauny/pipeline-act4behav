import argparse
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
import re

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
cutoff_hz = args.cutoff_hz
freq_measure = args.freq_measure
features_selection = args.features_selection
processes = args.processes

current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)

path = os.path.join(home_directory, id_expe, 'raw')
all_files = glob.glob(path+ "/*.csv")

behavior1 = args.behavior1
behavior2 = args.behavior2
behavior3 = args.behavior3
behavior4 = args.behavior4

behaviours = [behavior1,behavior2,behavior3,behavior4]
behaviours_names = '_'.join(behaviours)


print('Application filter')
print('Nombre de fichiers sélectionnés =', len(all_files) )
print('Files =', all_files)
    

def application_rotation (type) :
    if type == 'median' :
        multivariate_median = np.median(raw_data[['ACCxf', 'ACCyf', 'ACCzf']], axis=0)
    if type == 'mean' :
        multivariate_median = np.mean(raw_data[['ACCxf', 'ACCyf', 'ACCzf']], axis=0)
    target_direction = np.array([0, 0, 1])
    # Compute the rotation axis and angle
    rotation_axis = np.cross(multivariate_median, target_direction)
    rotation_angle = np.arccos(np.dot(multivariate_median, target_direction) / (np.linalg.norm(multivariate_median) * np.linalg.norm(target_direction)))
    
    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(rotation_angle) + rotation_axis[0]**2 * (1 - np.cos(rotation_angle)),
                                rotation_axis[0] * rotation_axis[1] * (1 - np.cos(rotation_angle)) - rotation_axis[2] * np.sin(rotation_angle),
                                rotation_axis[0] * rotation_axis[2] * (1 - np.cos(rotation_angle)) + rotation_axis[1] * np.sin(rotation_angle)] ,
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(rotation_angle)) + rotation_axis[2] * np.sin(rotation_angle),
                                np.cos(rotation_angle) + rotation_axis[1]**2 * (1 - np.cos(rotation_angle)),
                                rotation_axis[1] * rotation_axis[2] * (1 - np.cos(rotation_angle)) - rotation_axis[0] * np.sin(rotation_angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(rotation_angle)) - rotation_axis[1] * np.sin(rotation_angle),
                                rotation_axis[2] * rotation_axis[1] * (1 - np.cos(rotation_angle)) + rotation_axis[0] * np.sin(rotation_angle),
                                np.cos(rotation_angle) + rotation_axis[2]**2 * (1 - np.cos(rotation_angle))]])
    
    # Compute the inverse rotation matrix
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    
    # Apply the inverse rotation matrix to all accelerometer data
    rotated_data = np.dot(raw_data[['ACCxf', 'ACCyf', 'ACCzf']], inverse_rotation_matrix.T)
    
    # Add the rotated data columns to the raw_data DataFrame
    raw_data['ACCxf_r'] = rotated_data[:, 0]
    raw_data['ACCyf_r'] = rotated_data[:, 1]
    raw_data['ACCzf_r'] = rotated_data[:, 2]

def find_behavior_columns(raw_data, specified_behaviors, behavior_columns):
    relevant_columns = []
    for behavior in specified_behaviors:
        for column in behavior_columns:
            if behavior in raw_data[column].unique():
                relevant_columns.append(column)
                break  
    return relevant_columns

if filter == 'yes' :
    files_to_filter = []

    # Nyquist rate
    nyq_rate = freq_measure / 2.
    cutoff_hz=float(cutoff_hz)
    # Creation of the high pass Butterworth filter
    b, a = signal.butter(6, cutoff_hz/nyq_rate, 'high', analog=False)
    
    for file in all_files : 

        # Csv file opening
        raw_data = pd.read_csv(file,sep=",") 
        raw_data=raw_data.dropna(subset=['ACCx', 'ACCy', 'ACCz'])
        idgoat=str(raw_data["idgoat"][0])

        behavior_columns = raw_data.columns.tolist() 
        specified_behaviors = [behavior for behavior in [args.behavior1, args.behavior2, args.behavior3, args.behavior4] if behavior]
        behavior_related_columns = find_behavior_columns(raw_data, specified_behaviors, behavior_columns)

        # Filter application
        raw_data=raw_data.assign(ACCxf = signal.filtfilt(b, a, raw_data['ACCx']))
        raw_data=raw_data.assign(ACCyf = signal.filtfilt(b, a, raw_data['ACCy']))
        raw_data=raw_data.assign(ACCzf = signal.filtfilt(b, a, raw_data['ACCz']))

        if reference_vector_computation == 'median':
            application_rotation('median')
            raw_data=raw_data.dropna(subset=['ACCxf_r', 'ACCyf_r', 'ACCzf_r'])
            selected_columns = ['TIME','ACCxf', 'ACCyf', 'ACCzf','ACCxf_r', 'ACCyf_r', 'ACCzf_r','idgoat']
            selected_columns.extend(behavior_related_columns)

        if reference_vector_computation == 'mean' :
            application_rotation('mean')
            raw_data=raw_data.dropna(subset=['ACCxf_r', 'ACCyf_r', 'ACCzf_r'])
            selected_columns = ['TIME','ACCxf', 'ACCyf', 'ACCzf','ACCxf_r', 'ACCyf_r', 'ACCzf_r','idgoat']
            selected_columns.extend(behavior_related_columns)
        reference_vector_computation_file = "_"+reference_vector_computation
        if reference_vector_computation=="" :
            selected_columns = ['TIME','ACCxf', 'ACCyf', 'ACCzf','idgoat']
            selected_columns.extend(behavior_related_columns)

        missing_columns = [col for col in selected_columns if col not in raw_data.columns]

        if missing_columns:
            selected_columns = [col for col in selected_columns if col not in behavior_related_columns]

        # Select columns from raw_data
        raw_data = raw_data[selected_columns]


        # Write new csv    
        if euler_angles=="yes" :
            euler_angles='euler_angles'
        if euler_angles=="no" :
            euler_angles=''
        if filter=="yes" :
            filter="filter"+str(cutoff_hz)
        if filter=="no" :
            filter="no_filter"
        time_windows=str(time_windows)+"s"
        features_selection=str(features_selection)

        path_folder=home_directory+'/'+id_expe+"/filtered/"    
        components = [id_expe, idgoat, filter] 
        if reference_vector_computation: 
            components.insert(2, reference_vector_computation)  
        filepath = Path(path_folder + "_".join(components) + ".csv")

        # Check if the folder and the output file already exists
        if not os.path.exists(path_folder):
            os.makedirs(path_folder, exist_ok=True)      
        if filepath.exists():
            files_to_filter.append(file)
            continue

        filepath.parent.mkdir(parents=True, exist_ok=True)  
        raw_data.to_csv(filepath) 
   

if filter=='no' :
    files_to_filter = []
    for file in all_files : 
        # Csv file opening
        raw_data = pd.read_csv(file,sep=",") 
        raw_data=raw_data.dropna(subset=['ACCx', 'ACCy', 'ACCz'])
        idgoat=str(raw_data["idgoat"][0])

        behavior_columns = raw_data.columns.tolist() 
        specified_behaviors = [behavior for behavior in [args.behavior1, args.behavior2, args.behavior3, args.behavior4] if behavior]
        behavior_related_columns = find_behavior_columns(raw_data, specified_behaviors, behavior_columns)


        # Rename columns
        raw_data = raw_data.rename(columns={'ACCx': 'ACCxf', 'ACCy': 'ACCyf', 'ACCz': 'ACCzf'})

        if reference_vector_computation == 'median':
            application_rotation('median')
            raw_data=raw_data.dropna(subset=['ACCxf_r', 'ACCyf_r', 'ACCzf_r'])
            selected_columns = ['TIME','ACCxf', 'ACCyf', 'ACCzf','ACCxf_r', 'ACCyf_r', 'ACCzf_r','idgoat']
            selected_columns.extend(behavior_related_columns)

        if reference_vector_computation == 'mean' :
            application_rotation('mean')
            raw_data=raw_data.dropna(subset=['ACCxf_r', 'ACCyf_r', 'ACCzf_r'])
            selected_columns = ['TIME','ACCxf', 'ACCyf', 'ACCzf','ACCxf_r', 'ACCyf_r', 'ACCzf_r','idgoat']
            selected_columns.extend(behavior_related_columns)

        if reference_vector_computation=="" :
            selected_columns = ['TIME','ACCxf', 'ACCyf', 'ACCzf','idgoat']
            selected_columns.extend(behavior_related_columns)
        missing_columns = [col for col in selected_columns if col not in raw_data.columns]

        if missing_columns:
            selected_columns = [col for col in selected_columns if col not in behavior_related_columns]
            
        raw_data = raw_data[selected_columns]

        # Write new csv    
        if euler_angles=="yes" :
            euler_angles='euler_angles'
        if euler_angles=="no" :
            euler_angles=''
        if filter=="yes" :
            filter="filter"+str(cutoff_hz)
        if filter=="no" :
            filter="no_filter"
        time_windows=str(time_windows)+"s"
        features_selection=str(features_selection)
        
        path_folder=home_directory+'/'+id_expe+"/filtered/"    
        components = [id_expe, idgoat, filter, behaviours_names]  
        if reference_vector_computation:  
            components.insert(2, reference_vector_computation)  
        filepath = Path(path_folder + "_".join(components) + ".csv")

        # Check if the output file already exists
        if not os.path.exists(path_folder):
            os.makedirs(path_folder, exist_ok=True)
            
        if filepath.exists():
            files_to_filter.append(file)
            continue

        filepath.parent.mkdir(parents=True, exist_ok=True)  
        raw_data.to_csv(filepath)

if files_to_filter:
    print(f"The output files already exist. Skipping filtering for the {len(files_to_filter)} files.")
     
