import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name="gpheat"

list_of_files = [
    "config/config.yaml",
    "reports/metrics.json",
    "reports/figures/prediction_test.png",
    "dvc.yaml",
    "pyproject.toml",
    "requirements.txt",
    "research/trials.ipynb",
    f"src/{project_name}/cli.py",
    f"src/{project_name}/pipeline.py",    
    f"src/{project_name}/config.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/loaders.py",
    f"src/{project_name}/simulate/pybamm_model.py",
    f"src/{project_name}/models/kernels.py",
    f"src/{project_name}/models/gp.py",          
    f"src/{project_name}/evaluation/metrics.py",          
    f"src/{project_name}/plot/series.py",          
    f"src/{project_name}/experiments/sensitivity.py",
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} is already exists")
