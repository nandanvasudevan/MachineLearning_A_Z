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
from sklearn.linear_model import LinearRegression

# Read dataset | Nandan V - Sat 02/Jan/2021 23:54::43
dataset = pd.read_csv("../Data/Salary_Data.csv", header=0)
print("Dataset\n", dataset)

salaries = dataset.iloc[:, -1].values
print("\nSalary\n", salaries)

experiences = dataset.iloc[:, :-1].values
print("\nExperience\n", experiences)

(
    experiences_trainingSet,
    experiences_testingSet,
    salaries_trainingSet,
    salaries_testingSet,
) = train_test_split(experiences, salaries, test_size=0.2, random_state=0)

print(
    f"\nExperiences Training Set: \n{experiences_trainingSet} \n\nExperiences testing set: \n{experiences_testingSet}"
)
print(
    f"\nSalaries Training Set: \n{salaries_trainingSet} \n\nSalaries testing set: \n{salaries_testingSet}"
)

# Making a linear regression model | Nandan V - Sun 03/Jan/2021 10:00::57
linearRegressor = LinearRegression()

# Train the model | Nandan V - Sun 03/Jan/2021 10:02::52
experiences_trainingSet = experiences_trainingSet.reshape(-1, 1)
salaries_trainingSet = salaries_trainingSet.reshape(-1, 1)
experiences_testingSet = experiences_testingSet.reshape(-1, 1)


# Fir & predict using the model | Nandan V - Sun 03/Jan/2021 10:37::29
linearRegressor.fit(experiences_trainingSet, salaries_trainingSet)
prediction_testSet = linearRegressor.predict(experiences_testingSet)
prediction_trainingSet = linearRegressor.predict(experiences_trainingSet)

# Visualize the output | Nandan V - Sun 03/Jan/2021 10:37::47
plt.subplot(1, 2, 1)
plt.title("Training set")
plt.scatter(experiences_trainingSet, salaries_trainingSet, color="red")
plt.plot(experiences_trainingSet, prediction_trainingSet)
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.subplot(1, 2, 2)
plt.title("Testing set")
plt.plot(experiences_testingSet, prediction_testSet)
plt.scatter(experiences_testingSet, salaries_testingSet, color="green", alpha=0.4)
plt.scatter(experiences_testingSet, prediction_testSet, marker="+")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Testing result
print("\n\nPredicted Results")
print("-----------------")
print(f"\nExperience\tActual salary\tPredicted salary\n")
predictedSet = zip(experiences_testingSet, salaries_testingSet, prediction_testSet)
for experience, salary, predicted in predictedSet:
    print(f"{experience}\t\t{salary}\t\t{predicted}")

# Single sample prediction | Nandan V - Sun 03/Jan/2021 11:06::52
experience = 4.95
print(
    f"\nPrediction for {experience} year(s): \t{linearRegressor.predict([[experience]])}"
)
