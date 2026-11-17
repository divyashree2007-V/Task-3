
# TASK 3 - EXPLORATORY DATA ANALYSIS
# Student Performance Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders
os.makedirs("visualizations", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1. LOAD DATASET
data = pd.read_csv("student-mat.csv", sep=";")

print("\n===== DATASET OVERVIEW =====")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nData Types:\n", data.dtypes)

# 2. DATA CLEANING
print("\n===== DATA CLEANING =====")

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:", data.duplicated().sum())

data = data.drop_duplicates().copy()

# Handle missing values
numeric_cols = data.select_dtypes(include=np.number).columns
categorical_cols = data.select_dtypes(include="object").columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

data.to_csv("reports/cleaned_student_data.csv", index=False)

# 3. STATISTICAL SUMMARY
print("\n===== STATISTICAL SUMMARY =====")

print(data.describe())

print("\nFinal Grade Mean:", data["G3"].mean())
print("Final Grade Median:", data["G3"].median())
print("Final Grade Mode:", data["G3"].mode()[0])

data.describe().to_csv("reports/statistical_summary.csv")

# 4. OUTLIER DETECTION
print("\n===== OUTLIER DETECTION =====")

outlier_counts = {}

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((data[col] < lower) | (data[col] > upper)).sum()
    outlier_counts[col] = count

print(pd.Series(outlier_counts))

# 5. HISTOGRAM
plt.figure(figsize=(8, 5))
sns.histplot(data["G3"], bins=10, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.savefig("visualizations/grade_distribution.png")
plt.show()

# 6. BAR CHART
plt.figure(figsize=(8, 5))
sns.barplot(data=data, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Average Final Grade")
plt.savefig("visualizations/studytime_vs_grade.png")
plt.show()

# 7. BOX PLOT
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="failures", y="G3")
plt.title("Past Failures vs Final Grade")
plt.xlabel("Past Failures")
plt.ylabel("Final Grade")
plt.savefig("visualizations/failures_vs_grade.png")
plt.show()

# 8. SCATTER PLOT
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x="absences", y="G3")
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("Final Grade")
plt.savefig("visualizations/absences_vs_grade.png")
plt.show()

# 9. CORRELATION ANALYSIS
print("\n===== CORRELATION ANALYSIS =====")

correlation = data.select_dtypes(include=np.number).corr()

print(correlation["G3"].sort_values(ascending=False))

plt.figure(figsize=(12, 9))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()

correlation.to_csv("reports/correlation_matrix.csv")

# 10. IMPORTANT FACTORS
factors = correlation["G3"].drop("G3")
top_factors = factors.abs().sort_values(ascending=False).head(5)

print("\n===== TOP 5 FACTORS =====")
print(top_factors)

# 11. GENERATE EDA REPORT
with open("reports/EDA_Report.txt", "w") as file:

    file.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    file.write("STUDENT PERFORMANCE ANALYSIS\n\n")

    file.write("1. Dataset Overview\n")
    file.write(f"Total Records: {data.shape[0]}\n")
    file.write(f"Total Columns: {data.shape[1]}\n\n")

    file.write("2. Statistical Summary\n")
    file.write(str(data.describe()))

    file.write("\n\n3. Final Grade Analysis\n")
    file.write(f"Mean: {data['G3'].mean():.2f}\n")
    file.write(f"Median: {data['G3'].median():.2f}\n")
    file.write(f"Mode: {data['G3'].mode()[0]}\n")

    file.write("\n\n4. Outlier Analysis\n")
    file.write(str(pd.Series(outlier_counts)))

    file.write("\n\n5. Top Factors Associated with Final Grade\n")
    file.write(str(top_factors))

    file.write("\n\n6. Conclusion\n")
    file.write(
        "The analysis explores student grades, study time, "
        "past failures and absences. Correlation analysis "
        "helps identify relationships between these variables."
    )

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")
print("Cleaned dataset saved.")
print("Visualizations saved.")
print("EDA report generated.")
