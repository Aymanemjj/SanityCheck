import pandas as pd
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
import os

quantitative = {
    "continuous":["Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Social_Media_Hours", "Screen_Time_Hours", "Financial_Stress", "Family_Support", "Attendance_Rate"],
    "discrete":["Age", "Year_of_Study", ],
}
qualitative = {
    "nominal":["Gender", "Part_Time_Job", "Previous_Mental_Health_History"],
    "ordinal":["Physical_Activity", "Diet_Quality", "Mental_Health_Risk"],
    "ordinal_order":{
        "Physical_Activity":["Low", "Moderate", "High"],
        "Diet_Quality":["Poor", "Average", "Good"],
        "Mental_Health_Risk":["Low", "Medium", "High"],
    }
}


def check_data():
    df = pd.read_csv("student_mental_health.csv")
    
    print(f"Data shape: {df.shape}")
    print(f"Nulls:\n{df.isnull().sum()}\n")
    print(f"Total nulls:\n {df.isnull().sum().sum()}\n\n")
    print(f"Duplicates:\n{df.duplicated().sum()}\n\n")

def gotta_catch_them_all(data):
    data = data.copy()
    for col in data.columns:
        z = sp.stats.zscore(data[col], nan_policy="omit")
        inliers = data.loc[np.abs(z) < 3, col]
        data[col] = data[col].clip(inliers.min(), inliers.max())
    return data

def prepare_data():
    df = pd.read_csv("student_mental_health.csv")
    df = df.drop(columns="Student_ID")
    df = df.drop_duplicates()

    X = df.drop(columns="Mental_Health_Risk")
    Y = df["Mental_Health_Risk"]
    Y = Y.map({"Low":0, "Medium":1, "High":2})
    

    X[quantitative["continuous"] + quantitative["discrete"]] = gotta_catch_them_all(X[quantitative["continuous"] + quantitative["discrete"]])

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    print(os.getcwd())
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    Y_train.to_csv("data/Y_train.csv", index=False)
    Y_test.to_csv("data/Y_test.csv", index=False)
