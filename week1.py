import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Analyzing the dataset, we have an unwanted (last) column, therefore:
columnsToInclude = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

#Also, we want to avoid any NaN values, so we do:
df = pd.read_csv('data.csv', sep = ",", usecols=columnsToInclude).fillna(0)

print(df.info())

print(df.head())

#Changing the values of M to 0 and B to 1 to apply logistic regression:
for index, value in df.iterrows():
    if value['diagnosis'] == "M":
        df.at[index, 'diagnosis'] = 0
    else:
        df.at[index, 'diagnosis'] = 1

print(df.head())

#Attempt for PCA using the more "meaningful" parts of the dataset:
X = df[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"]]

pca = PCA()

pca.fit(X)

varianceRatio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(varianceRatio)

principalComponents = pca.components_
print("Principal Components:")
print(principalComponents)

XPca = pca.transform(X)
print(XPca)

#We now have the PCA values for our data, which is more "compact" than before

#Logistic regression portion of the code:
#We already have the values for our features, which will be X_pca
YVar = df['diagnosis'] 
YVar = YVar.astype('int')

XTrain, XTest, YTrain, YTest = train_test_split(XPca, YVar)
model = LogisticRegression()
model.fit(XTrain, YTrain)

YPred = model.predict(XTest)

accuracy = accuracy_score(YTest, YPred)
print("Accuracy:", accuracy)