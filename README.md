<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/smauny/pipeline-modbehav/tree/main/images">
    <img src="https://github.com/smauny/pipeline-modbehav/blob/c55afed3d5eb781c031967b531ea6a22ec0dec1c/images/Modbehav_logo.png" alt="Logo" width="400" height="300">
  </a>

<h3 align="center">ModBehav: a Machine Learning pipeline with extensive pre-processing and feature creation options</h3>

  <p align="center">
    Python scripts pipeline to detect animal behaviours from accelerometer data 
    <br />
    <a href="https://github.com/smauny/pipeline-modbehav/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/smauny/pipeline-modbehav/">Report Bug</a>
    
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#more">More</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This GitHub repository contains a set of Python scripts presented in X to automatically pre-process data and train classification models to detect animal behaviours from accelerometer data. A set of example data is available on: X and is presented in the following datapaper: X.

This README file provides a step by step guide on how to use this pipeline. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.js]][Python-url]
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
### Prerequisites

* Python packages
  ```sh
  pip install numpy pandas plotly glob2 scikit-learn catboost seaborn matplotlib tsfresh
  ```

### Installation

Clone the repo in your working folder
   ```sh
   git clone https://github.com/smauny/pipeline-modbehav.git
   ```

### Available data

You can download example data on: 
This data is presented in our datapaper:

Save the raw data in your working directory in a folder.

<!-- USAGE EXAMPLES -->
## Usage
### Basic run 
In your terminal, you can start by running the scripts_pipeline.py file

   ```sh
   python3 /your/working/directory/scripts/scripts_pipeline.py
   ```


Several questions will be asked. You can answer the questions one by one by following the instructions in the parenthesis and click enter to answer the next question

- **What is the name of the data folder?** Enter the name of the folder in your working directory where your raw data is located.
- **Do you want to filter the data?** Enter yes or no to apply a high-pass filter on the raw acceleration data.
- **What type of reference vector computation do you want? (median/mean/blank)** Enter the name of the transformation you want to apply to your data. This will create an additional time-serie in your dataset.
- **What is the cutoff frequency (X.x Hz/blank)?** If you entered 'yes' for the data filter, enter the value of the cutoff frequency.
- **What is the frequency measurement (X.x Hz)?** Enter the sampling rate of the accelerometer data in Hertz.
- **What is the size of the time windows (s)?** Enter the size of the time-windows you want to use in seconds to segmentate the data
- **Do you want to include Euler angles? (yes/no)** Enter yes or no if you want to include or not the pitch and roll time-series calculated from your data/
- **How many of the best features do you want to select (x/blank)** Enter the number of the best features you want to keep (features selection)
- **First behavior to predict?
Second behavior to predict?
Third behavior to predict?
Fourth behavior to predict?** Enter the names of the behaviours you want to detect. The names should be the same as in your data.

<br />
<div align="center">
  <a href="https://github.com/smauny/pipeline-modbehav/tree/main/images">
    <img src="https://github.com/smauny/pipeline-modbehav/blob/fdc9a84e1805fee04cc3a677e91f54f18532f0ff/images/Screenshot_scripts_pipeline.png" alt="Screenshot" width="600">
  </a>
</div>



### Multiple runs
If you want to execute multiple runs automatically, you can open and modify the control_script_pipeline.py script instead of the scripts_pipeline.py. Enter directly the parameters you want to choose for your runs. In this example, multiple time-windows sized are tested (10, 20, 30 and 40 seconds)

   ```sh
   pipeline_runs = [
    {"id_expe": "adapt", "filter": "no", "reference_vector_computation": "mean", "cutoff_hz": "", "freq_measure": "5", "time_windows": "10", "euler_angles": "yes", "features_selection": ""},
    {"id_expe": "adapt", "filter": "no", "reference_vector_computation": "mean", "cutoff_hz": "", "freq_measure": "5", "time_windows": "20", "euler_angles": "yes", "features_selection": ""},
    {"id_expe": "adapt", "filter": "no", "reference_vector_computation": "mean", "cutoff_hz": "", "freq_measure": "5", "time_windows": "30", "euler_angles": "yes", "features_selection": ""},
    {"id_expe": "adapt", "filter": "no", "reference_vector_computation": "mean", "cutoff_hz": "", "freq_measure": "5", "time_windows": "40", "euler_angles": "yes", "features_selection": ""}
]
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

Save the code and run the control_script_pipeline.py instead of the scripts_pipeline.py

   ```sh
   python3 /your/working/directory/scripts/control_script_pipeline.py
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Output
#### 1. You provided training data (acceleration data + associated behaviours) 
The pipeline pre-processes your data and trains four models for the behaviours that you chose.
The pipeline creates in output the pre-processed data in a .csv file in the folder "pre-processed data".
The pipeline creates in output in the folder "scores_features_models" (automatically created in your working directory):
- the scores of the trained models in .csv files
- the trained models in .cbm files
- the features importances of each model on .csv files
- the features selected for each model (depending on the features selection parameter applied) on .txt files

#### 2. You provided acceleration data 
The pipeline pre-processes your data and detects automatically that you don't want to train models. 
The pipeline creates in output the pre-processed data in a .csv file in the folder "pre-processed data".

<!-- MORE -->
## More

To have more details on this work, please read our paper:
And datapaper: 

<!-- CONTACT -->
## Contact

Mauny Sarah - sarah.mauny@inrae.fr

Project Link: [https://github.com/smauny/pipeline-modbehav](https://github.com/smauny/pipeline-modbehav)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Python.js]: https://img.shields.io/pypi/pyversions/scikit-learn
[Python-url]: https://pypi.org/project/Python.js/

