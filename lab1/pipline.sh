#!/bin/bash

pip install -r requirements.txt

python3 1_data_creation.py
python3 2_data_preprocessing.py
python3 3_model_preparation.py
python3 4_model_testing.py