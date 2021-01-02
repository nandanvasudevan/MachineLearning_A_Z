"""
File:   simpleRegression
Author: Nandan V (nvpotti.mec@gmail.com)
Date:   Sat 02/Jan/2021 23:29::44
Implementation for simple regression using salary dataset.
"""

# Importing libraries | Nandan V - Sat 02/Jan/2021 23:33::37
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import pandas as pd
from sklearn.model_selection import train_test_split

# Read dataset | Nandan V - Sat 02/Jan/2021 23:54::43
dataset = pd.read_csv("../Data/Salary_Data.csv", header=0)
# Converting salary to int32
dataset = pd.DataFrame(data=dataset)
dataset = dataset.astype({"Salary": "int32"})
print("Dataset\n", dataset)

salaries = dataset.iloc[:, -1]
print("\nSalary\n", salaries)

experiences = dataset.iloc[:, 0]
print("\nExperience\n", experiences)

(
    experiences_trainingSet,
    experiences_testingSet,
    salaries_trainingSet,
    salaries_testingSet,
) = train_test_split(experiences.values, salaries.values, test_size=0.2, random_state=0)

print(
    f"Experiences Training Set: \n{experiences_trainingSet} \nExperiences testing set: \n{experiences_testingSet}"
)
print(
    f"Salaries Training Set: \n{salaries_trainingSet} \nSalaries testing set: \n{salaries_testingSet}"
)