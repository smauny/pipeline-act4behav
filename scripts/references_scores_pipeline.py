import numpy as np
import pandas as pd
import plotly.express as px
import glob
import os.path
import datetime
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from math import*
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostError
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sys
import re
import argparse
import csv

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
behavior1 = args.behavior1
behavior2 = args.behavior2
behavior3 = args.behavior3
behavior4 = args.behavior4
processes=args.processes

current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)

behaviours = [behavior1,behavior2,behavior3,behavior4]
behaviours_names = '_'.join(sorted(behaviours))

features_parameters = "EfficientFCParameters"
if reference_vector_computation=='' :
    reference_vector_computation2=''
if reference_vector_computation!='' : 
    reference_vector_computation2="_"+reference_vector_computation


if filter=='yes' :
    filter='filter'
    cut_off_freq_str = (re.sub(r'[^\w\s]', '', str(cutoff_hz)))+"hz_"
if filter=='no' :
    filter='no_filter'
    cut_off_freq_str = ""

if euler_angles=='yes' :
    euler_angles='_euler_angles'
if euler_angles=='no' :
    euler_angles=''

path=home_directory+'/preprocessed_data/'+id_expe+reference_vector_computation2+'_'+filter+'_'+cut_off_freq_str+str(time_windows)+'s'+'_'+behaviours_names+'_'+features_parameters+euler_angles+'.csv'

print('References scores script')
print('File =', path)
dataset = pd.read_csv(path,sep=",")
filtered_columns = dataset.filter(like='Unnamed:', axis=1)
dataset = dataset.drop(filtered_columns.columns, axis=1)
dataset = dataset.drop_duplicates(subset=['start_window_id'])

dataset = dataset.dropna(axis=1, how='any')

features = dataset.iloc[:,1:dataset.shape[1]-10] #10=nb of labels (at the end of the df)
labels = dataset.filter(like='yes_no').columns.tolist()

auc_scores = []
accuracy_scores = []
balanced_accuracy_scores = []
f1_scores = []
specificity_scores = []
sensitivity_scores = []


output_file_name_parts = [id_expe, reference_vector_computation2, filter, cut_off_freq_str, str(time_windows) + 's', behaviours_names, euler_angles, features_selection]
output_file_name = '_'.join(part for part in output_file_name_parts if part)
    
output_file = f"{output_file_name}_scores.csv"
output_path=Path(home_directory+'/scores_features_models/'+output_file)

# Check if the output folder already exists
path_folder=home_directory+'/scores_features_models/'
if not os.path.exists(path_folder):
    os.makedirs(path_folder, exist_ok=True)

# Check if the output files already exist
if output_path.exists() :
    print(f"The output files {output_path} already exist. Skipping scores calculation.")
    exit()

def write_scores_to_csv(output_file, scores):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'AUC', 'Accuracy', 'Balanced Accuracy', 'F1-score', 'Sensitivity', 'Specificity'])
        for label, auc, accuracy, balanced_accuracy, f1, sensitivity, specificity in scores:
            writer.writerow([label, auc, accuracy, balanced_accuracy, f1, sensitivity, specificity])

scores=[]

with open(output_path, "w") as f:
    for label in labels :

        features_all = features.columns.tolist()

        # Get  features
        features_list = features_all


        # ML for all labels
        X=dataset[features_list]  # Features
        y=dataset[label]  # Labels
        y=y.astype('int')

        # Split dataset into training set and test set
        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

        # Split the training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=42)
        n=4000
        # early_stopping=(10/100)*n
        #Create a Gradient boosting classifier
        try:
            clf=CatBoostClassifier(learning_rate=.003, od_type='Iter', n_estimators=n,task_type="GPU", verbose=False,class_weights={0: 1, 1: 2},random_strength= 0, depth=8, l2_leaf_reg=2,gpu_ram_part=0.9, devices='0:1')
            #Train the model using the training sets y_pred=clf.predict(X_test)
            clf.fit(X_train,y_train,eval_set=(X_val,y_val), use_best_model=True)
            
        except Exception as e:
            clf=CatBoostClassifier(learning_rate=.003, od_type='Iter', n_estimators=n,task_type="CPU", verbose=False,class_weights={0: 1, 1: 2},random_strength= 0, depth=8, l2_leaf_reg=2)
            #Train the model using the training sets y_pred=clf.predict(X_test)
            clf.fit(X_train,y_train,eval_set=(X_val,y_val), use_best_model=True)

        y_pred=clf.predict_proba(X_test)

        # Features importance
        fi = pd.DataFrame({'name': features_list, 'w': clf.feature_importances_})
        fi = fi.sort_values('w')


        if features_selection!='' :
            n=64000
            fi = pd.DataFrame({'name':features_list,'w':clf.feature_importances_})
            fi=fi.sort_values('w')
            t=len(fi)-int(features_selection)
            #X_fi=fi[int(t*len(fi)):len(fi)]  # Features
            X_fi=fi[t:len(fi)]
            X_fi_list=X_fi['name'].values.tolist()
            X=dataset[X_fi_list]
            y=dataset[label]  # Labels
            y=y.astype('int')
            print('Nombre de features X_fi =', len(X_fi))
            X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
            X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=42)
            
            try :
                clf=CatBoostClassifier(learning_rate=.001, od_type='Iter', n_estimators=n,task_type="GPU",verbose=False,class_weights={0: 1, 1: 2},random_strength= 0, depth=9, l2_leaf_reg=2,gpu_ram_part=0.9, devices='0:1')
                clf.fit(X_train,y_train,eval_set=(X_val,y_val), use_best_model=True)

            except Exception as e:
                clf=CatBoostClassifier(learning_rate=.001, od_type='Iter', n_estimators=n,task_type="CPU",verbose=False,class_weights={0: 1, 1: 2},random_strength= 0, depth=9, l2_leaf_reg=2)
                clf.fit(X_train,y_train,eval_set=(X_val,y_val), use_best_model=True)

            y_pred=clf.predict_proba(X_test)

            selected_feature_names = X_fi_list

            # Save the trained CatBoost model
            model_path = home_directory+'/scores_features_models/'+label+output_file_name+'.cbm'
            clf.save_model(model_path)
            print(f"Model saved to {model_path}")

            # Save the table as a CSV file
            fi.to_csv(home_directory+'/scores_features_models/feature_importance_'+label+output_file_name+'.csv', index=False)


        print('Nombre de features =', len(X))
        print('Nombre de features fi =', len(fi))

        # AUC
        auc = roc_auc_score(y_test, y_pred[:,1])
        auc_scores.append((label, " - AUC = ",auc))
        print(label, " - AUC = ",auc)
        print(auc_scores)

        threshold = 0.5

        # Accuracy
        chosen_predictions = y_pred[:, 1] >= threshold
        accuracy = accuracy_score(y_test, chosen_predictions)
        accuracy_scores.append((label, " - Accuracy =", accuracy))
        print(label, " - Accuracy =", accuracy)

        accuracy_scores = []
        df=pd.DataFrame({'Prediction': y_pred[:, 1], 'label': y_test})
        # Iterate over different threshold values with a step of 0.01
        for threshold in np.arange(0.00, 1.00, 0.01):
            y_predictions = (df['Prediction'] > threshold).astype(int)
            accuracy = accuracy_score(df['label'], y_predictions)
            accuracy_scores.append(accuracy)

        # Convert the list to a numpy array for easy manipulation
        accuracy_array = np.array(accuracy_scores)
        best_threshold_index = np.argmax(accuracy_array)
        best_threshold = np.arange(0.00, 1.00, 0.01)[best_threshold_index]
        best_accuracy = accuracy_array[best_threshold_index]
       
        # Balanced Accuracy
        balanced_accuracy = balanced_accuracy_score(y_test, chosen_predictions)
        balanced_accuracy_scores.append((label, " - Balanced Accuracy =", balanced_accuracy))
        print(label, " - Balanced Accuracy =", balanced_accuracy)

        # F1-score
        f1 = f1_score(y_test, chosen_predictions)
        f1_scores.append((label, " - F1-score =", f1))
        print(label, " - F1-score =", f1)

        # Sensitivity (Recall)
        sensitivity = recall_score(y_test, chosen_predictions)
        sensitivity_scores.append((label, " - Sensitivity =", sensitivity))
        print(label, " - Sensitivity =", sensitivity)

        # Specificity
        specificity = precision_score(y_test, chosen_predictions)
        specificity_scores.append((label, " - Specificity =", specificity))
        print(label, " - Specificity =", specificity)

        f.write(f"{label}\n")
        f.write("AUC: {}\n".format(auc))
        f.write("Accuracy: {}\n".format(accuracy))
        f.write("Best Accuracy: {} (Threshold: {})\n".format(best_accuracy, best_threshold))
        f.write("Balanced Accuracy: {}\n".format(balanced_accuracy))
        f.write("F1-score: {}\n".format(f1))
        f.write("Sensitivity: {}\n".format(sensitivity))
        f.write("Specificity: {}\n".format(specificity))
        f.write("\n")

        scores.append((label, auc, accuracy, balanced_accuracy, f1, sensitivity, specificity))


    
print(auc_scores)
print(accuracy_scores)
print(balanced_accuracy_scores)
print(f1_scores) 
print(specificity_scores)
print(sensitivity_scores)



write_scores_to_csv(output_file, scores)

