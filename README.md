# Data and Process Science on the BPIC-17 Event Log
## IN0012: Business Process Prediction, Simulation, and Optimisation
## Description

This projects analyses the BPIC-17 event log, applying different process discovery algorithms, to create a custom model. This model is checked and verified using various quality metrics including fitness, precision, generalisation, and simplicity.

The entire analysis, including data loading, process discovery, metric calculation, and visualisation, is contained within the Jupyter Notebook.

## Getting started
This project requires a Python environment (preferably Conda) and the following external libraries with versions listed in `requirements.txt`.
### Dependencies
- Python 3.12.12
- Core libraries: pm4py, pandas, scikit-learn

### Installing
To set up the environment, clone the repository and use `pip` to install all dependencies:
```
pip install -r requirements.txt
```

### Executing program
1. Clone the repo
    ```
    git clone https://github.com/mbrieg/bppso-task1.git
    ```
- **Start the Jupyter Server:**
    ```
    jupyter notebook .
    ```

- **Open and Run:** Open `firstexercise.ipynb` and execute all cells separately or run all at once.

## Project Structure
```
.
├── firstexercise.ipynb         # Main analysis, execution, and results
├── final_model.bpmn           
├── helper.py                   # Custom Helper class 
├── requirements.txt            
├── BPI Challenge 2017_1_all/   # Directory for the event log file
├── process_discovery/          # Directory for models derived using Pm4Py
    ├── filtered_logs/
    ├── heuristic_nets/
├── process_models/             # Directory for custom models
    ├── Pm4Py_Visualisations/
└── README.md                   
```
