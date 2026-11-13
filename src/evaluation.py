import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, classification_report





def get_predections():
    model = joblib.load("models/LRP.joblib")

    X_test = pd.read_csv("data/X_test.csv")
    Y_test = pd.read_csv("data/Y_test.csv").iloc[:, 0]   

    Y_pred = model.predict(X_test)

    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.to_csv("data/Y_pred.csv", index=False)



def evaluate_model(params):
    label_map = {0: "Low", 1: "Medium", 2: "High"}

    Y_test = pd.read_csv("data/Y_test.csv").iloc[:, 0]
    Y_pred = pd.read_csv("data/Y_pred.csv").iloc[:, 0]
    Y_test_mapped = Y_test.map(label_map).reset_index(drop=True)
    Y_pred_mapped = pd.Series(Y_pred).map(label_map)

    ConfusionMatrixDisplay.from_predictions(Y_test_mapped, Y_pred_mapped, cmap="Blues").figure_.savefig("data/last_run_confusionMatrix.jpg")

    comp = pd.DataFrame({
        "Predicted": Y_pred_mapped,
        "Actual": Y_test_mapped,
    }, index=range(len(Y_test_mapped)), columns=["Predicted", "Actual"])
    print(comp)
    print("\n\n\n")
    print(classification_report(
            Y_test,
            Y_pred,
            target_names=["Low", "Medium", "High"]
    ))
