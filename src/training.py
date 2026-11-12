from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import joblib


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

imputable_cols = ["Study_Hours_Per_Week", "Year_of_Study", "GPA", "Sleep_Hours", "Social_Media_Hours", "Screen_Time_Hours", "Attendance_Rate"]
other_quant = [c for c in quantitative["continuous"] + quantitative["discrete"]
               if c not in imputable_cols]



def ginki_dama(data):
    data = data.copy()
    data["Start_Age"] = data["Age"] - data["Year_of_Study"].round()
    data["Social_Media_Hours"] = data["Social_Media_Hours"].clip(upper=data["Screen_Time_Hours"])
    return data

def ginki_names(transformer, input_features):
    return list(input_features) + ["Start_Age"]





# Pipe parts
quant_pipe = ColumnTransformer(
    transformers=[
        ("stage_1", IterativeImputer(max_iter=10, random_state=42), imputable_cols),
        ("stage_2", SimpleImputer(strategy="median"), other_quant),
    ],
    verbose_feature_names_out=False,
).set_output(transform="pandas")

great_quant_pipe = Pipeline(steps=[
    ("imputation", quant_pipe),
    ("ginki_dama", FunctionTransformer(ginki_dama, feature_names_out=ginki_names)),
    ("scaler", StandardScaler()),
])

nominal_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
ordinal_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])












def assembler():
    preprocessing = ColumnTransformer(
    transformers=[
        ("quant", great_quant_pipe, quantitative["continuous"] + quantitative["discrete"]),
        ("nominal", nominal_pipe, qualitative["nominal"]),
        ("ordinal", ordinal_pipe, ["Physical_Activity", "Diet_Quality"]),

    ],
    remainder="drop",)

    reg = LogisticRegression(l1_ratio=0, max_iter=200, random_state=0, solver='lbfgs')

    LRP = Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", reg)
    ])
    joblib.dump(LRP, "models/LRP.joblib")


def train_model():
    
    model = joblib.load("models/LRP.joblib")
    X_train = pd.read_csv("data/X_train.csv") 
    Y_train = pd.read_csv("data/Y_train.csv")

    model.fit(X_train, Y_train)

    joblib.dump(model, "models/LRP.joblib")





