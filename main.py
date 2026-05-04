#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#Phase:1
load_and_clean_data()
#Phase:2
run_level_1()
#Phase:3
run_level_1()
#Phase:4
MarkPredictor class
#Phase:5
run_level_2()
#Phase:6
bias_audit()
#Phase:7
cross_validate()
#Phase:8
predict_alex()
#Phase:extension(band 6)
neural_network_comparison()
