import argparse
import subprocess
import os

# Define the base directory as the user's directory
current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)

# Define the list of script names in the desired execution order
script_names = [
    "/scripts/application_filter_pipeline.py",
    "/scripts/window_tache_prediction_pipeline.py",
    "/scripts/euler_angles_pipeline.py",
    "/scripts/timseries_features_engineering_pipeline.py",
    "/scripts/references_scores_pipeline.py"
]

# Complete paths for each script
script_paths = [os.path.join(home_directory+ script_name) for script_name in script_names]


# Define the list of arguments for the entire pipeline
pipeline_arguments = [
    {"name": "id_expe", "question": "What is the name of the data folder?"},
    {"name": "filter", "question": "Do you want to filter the data? (yes/no)"},
    {"name": "reference_vector_computation", "question": "What type of reference vector computation do you want? (median/mean/blank)"},
    {"name": "cutoff_hz", "question": "What is the cutoff frequency (X.x Hz/blank)?"},
    {"name": "freq_measure", "question": "What is the frequency measurement (X.x Hz)?"},
    {"name": "time_windows", "question": "What is the size of the time windows (s)?"},
    {"name": "euler_angles", "question": "Do you want to include Euler angles? (yes/no)"},
    {"name": "features_selection", "question": "How many of the best features do you want to select (x/blank)"},
    {"name": "behavior1", "question": "First behavior to predict? "},
    {"name": "behavior2", "question": "Second behavior to predict? "},
    {"name": "behavior3", "question": "Third behavior to predict? "},
    {"name": "behavior4", "question": "Fourth behavior to predict? "}

]

# Parse command-line arguments
parser = argparse.ArgumentParser()
for arg in pipeline_arguments:
    parser.add_argument(f"--{arg['name']}", help=arg["question"], default=None)

args = parser.parse_args()

for script_path in script_paths:
    command = ["python3", script_path]
    for arg in pipeline_arguments:
        arg_value = getattr(args, arg["name"])
        command.extend([f"--{arg['name']}", arg_value])
        if arg_value:
            command.extend([f"--{arg['name']}", arg_value])
    
    subprocess.run(command, check=True)
