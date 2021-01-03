"""
File:   multipleRegression
Author: Nandan V (nvpotti.mec@gmail.com)
Date:   Sun 03/Jan/2021 17:58::22
Multiple regression example using 50_startups dataset
"""

import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import matplotlib.pyplot as plt

# Enable or disable debug output | Nandan V - Sun 03/Jan/2021 18:47::43
debug = False

# Import dataset | Nandan V - Sun 03/Jan/2021 18:47::56
dataset = pd.read_csv("../Data/50_Startups.csv", header=0)

# Extract data | Nandan V - Sun 03/Jan/2021 18:48::10
profits = dataset.iloc[:, -1]
rndExpenidtures = dataset.iloc[:, 0]
adminExpenditures = dataset.iloc[:, 1]
marketingExpenditures = dataset.iloc[:, 2]
states = dataset.iloc[:, 3]

if debug:
    print(f"\nRaw dataset\n{dataset}")
    print(f"\nR&D expenditures\n{rndExpenidtures}")
    print(f"\nAdmin expenditures\n{adminExpenditures}")
    print(f"\nMarketing expenditures\n{marketingExpenditures}")
    print(f"\nStates\n{states}")
    print(f"\nProfits\n{profits}")

# Convert to arrays | Nandan V - Sun 03/Jan/2021 18:48::25
profits = profits.values
rndExpenidtures = rndExpenidtures.values
adminExpenditures = adminExpenditures.values
marketingExpenditures = marketingExpenditures.values
states = states.values

# Check if there are NaN values in the dataset | Nandan V - Sun 03/Jan/2021 19:00::20
# I know for a fact that there aren't any NaN values which is why I haven't bothered with wasting tim writing imputers which are never executed.
if np.isnan(rndExpenidtures).any():
    print("R&D Expenditure has NaN values")
else:
    print("R&D Expenditure does not have NaN values")

if np.isnan(adminExpenditures).any():
    print("R&D Expenditure has NaN values")
else:
    print("R&D Expenditure does not have NaN values")

if np.isnan(marketingExpenditures).any():
    print("R&D Expenditure has NaN values")
else:
    print("R&D Expenditure does not have NaN values")

if np.isnan(states).any():
    print("R&D Expenditure has NaN values")
else:
    print("R&D Expenditure does not have NaN values")

if np.isnan(profits).any():
    print("R&D Expenditure has NaN values")
else:
    print("R&D Expenditure does not have NaN values")
