import pandas as pd

#Analyzing the dataset, we have an unwanted (last) column, therefore:
columnsToInclude = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

#Also, we want to avoid any NaN values, so we do:
df = pd.read_csv('data.csv', sep = ",", usecols=columnsToInclude).fillna(0)

print(df.head())