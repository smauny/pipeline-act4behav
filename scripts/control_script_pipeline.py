import subprocess
import os.path

# Arguments definition 
pipeline_runs = [
    {"id_expe": "adapt", "filter": "no", "reference_vector_computation": "mean", "cutoff_hz": "", "freq_measure": "5", "time_windows": "600", "euler_angles": "yes", "features_selection": ""},
    {"id_expe": "adapt", "filter": "no", "reference_vector_computation": "median", "cutoff_hz": "", "freq_measure": "5", "time_windows": "600", "euler_angles": "yes", "features_selection": ""}

]

current_working_directory = os.path.abspath(__file__)
scripts_position = current_working_directory.find('scripts')
home_directory = current_working_directory[:scripts_position].rstrip(os.sep)

# Path to the script that defines all the scripts of the pipeline
pipeline_script = home_directory+"/scripts/scripts_pipeline_chain.py"

for run in pipeline_runs:
    command = ["python3", pipeline_script]
    for arg, value in run.items():
        command.append(f"--{arg}")
        command.append(value)
    subprocess.run(command, check=True)

