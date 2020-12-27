import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler


def output(title, array):
    print("")
    print(title)
    if(array.ndim > 1):
        for i in range(len(array)):
            print("\t" + str(array[i]))
    else:
        print("\t" + str(array))


# Import dataset
dataset = pd.read_csv('../Data/Data.csv', header=0,
                      true_values=['Yes'], false_values=['No'])
print("Dataset")
print(dataset)

# Create feature matrix with all features
featureMatrix = dataset.iloc[:, :-1].values
output("Raw feature matrix", featureMatrix)

# Create dependency matrix
dependencyMatrix = dataset.iloc[:, -1].values
output("Raw dependency matrix", dependencyMatrix)

# Fill in missing data
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_mean.fit(featureMatrix[:, 1:])
featureMatrix[:, 1:] = imputer_mean.transform(featureMatrix[:, 1:])
output("Imputed feature matrix", featureMatrix)

# Encode categorical data
""" One hot encoding would create a column for each unique item (here, a country).
 Kind of like a the binary | 8 | 4 | 2 | 1 | representation of a hexadecimal number. """

# Encoding the feature matrix first
columnTransformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
featureMatrix_encoded = np.array(
    columnTransformer.fit_transform(featureMatrix))
output("One hot encoded feature matrix", featureMatrix_encoded)

"""  Since the values were converted to boolean while importing, they can be converted to 1s and 0s easily
 without a label encoder """
dependencyMatrix_labelled = np.multiply(dependencyMatrix, 1)
output("Labelled dependency matrix", dependencyMatrix_labelled)

# Split dataset into training and test sets
""" Feature scaling must be applied after splitting the data into training and test sets.
Feature scaling is done to normalize all the features so that one feature does not overpower any other.
Feature scaling must not be done on the test set since the test set must be kept as a brand new untouched set. This
might also lead to 'information leakage'. """

featureMatrix_trainingSet, featureMatrix_testSet, dependencyMatrix_trainingSet, dependencyMatrix_testSet = train_test_split(
    featureMatrix_encoded, dependencyMatrix_labelled, test_size=0.2, random_state=1)

output("Training set of feature matrix", featureMatrix_trainingSet)
output("Testing set of feature matrix", featureMatrix_testSet)
output("Training set of dependency matrix", dependencyMatrix_trainingSet)
output("Testing set of dependency matrix", dependencyMatrix_testSet)

# Feature scaling
""" There are two methods for scaling a feature.
1.  Standardization
    Here the feature is scaled to fit between [-3, 3].
    Works all the time.
    standardized_X = (x - mean(x))/stddev(x)
2. Normalization
    Here the feature is scaled to fit between [0 ,1].
    Recommended when there is normal distribution.
    norm_X = (x - min(x))/(max(x) - min(x))
3. Max-abs scaler
    Scales each feature by its maximum absolute value.
    This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
4. Using inter-quantile range (IQR)
    The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
    Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results.
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
"""
standardScaler = StandardScaler()
featureMatrix_trainingSet__stdScaled = featureMatrix_trainingSet
featureMatrix_trainingSet__stdScaled[:, 3:] = standardScaler.fit_transform(
    featureMatrix_trainingSet[:, 3:])
featureMatrix_testSet[:, 3:] = standardScaler.transform(
    featureMatrix_testSet[:, 3:])
output("Standardized feature training set", featureMatrix_trainingSet)
output("Standardized feature test set", featureMatrix_testSet)

minMaxScaler = MinMaxScaler()
featureMatrix_trainingSet__minMaxScaled = featureMatrix_trainingSet
featureMatrix_trainingSet__minMaxScaled[:, 3:] = minMaxScaler.fit_transform(
    featureMatrix_trainingSet[:, 3:])
output("Min max scaled feature training set",
       featureMatrix_trainingSet__minMaxScaled)

maxAbsScaler = MaxAbsScaler()
featureMatrix_trainingSet__maxAbsScaled = featureMatrix_trainingSet
featureMatrix_trainingSet__maxAbsScaled[:, 3:] = maxAbsScaler.fit_transform(
    featureMatrix_trainingSet[:, 3:])
output("Max abs scaled feature training set",
       featureMatrix_trainingSet__maxAbsScaled)

robustScaler = RobustScaler()
featureMatrix_trainingSet__robustScaled = featureMatrix_trainingSet
featureMatrix_trainingSet__robustScaled[:, 3:] = robustScaler.fit_transform(
    featureMatrix_trainingSet[:, 3:])
output("Robust scaler scaled feature training set",
       featureMatrix_trainingSet__robustScaled)
