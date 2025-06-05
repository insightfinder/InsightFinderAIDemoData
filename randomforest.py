import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from psi import calculate_psi

class RandomForestDrift:
    def __init__(self, n_estimators=410, random_state=42, max_depth=4):
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            random_state=random_state,
                                            max_depth=max_depth,)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def data_quality_injection(self, X_test, column_name="expenditure"):
        X_test_drift = X_test.copy()
        X_test_drift[column_name] = np.nan
        # X_test_drift['share'] = np.nan
        return X_test_drift

    def data_drift_injection(self, X_test, column_name="expenditure"):
        X_test_drift = X_test.copy()
        # Horizontally mirror the 'expenditure' column
        max_value = X_test_drift[column_name].max()
        X_test_drift[column_name] = max_value - X_test_drift[column_name]
        return X_test_drift

    def show_psi(self, y_pred, y_pred_drift):
        y_pred = np.vectorize({False: 0, True: 1}.get)(y_pred)
        y_pred_drift = np.vectorize({False: 0, True: 1}.get)(y_pred_drift)
        psi_diff = calculate_psi(y_pred, y_pred_drift, buckets=2, axis=1)
        print("PSI Difference:", psi_diff)

    def show_confusion_matrix(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)


class DataPreparation:
    def __init__(self, filename, ycolumn="card"):
        self.filename = filename
        self.ycolumn = ycolumn

    def load_dataset_file(self):
        # Load the dataset
        df = pd.read_csv(self.filename, sep=",")
        # Transform into Numerical Data
        df = pd.get_dummies(data=df, drop_first=True)
        transformed_ycolumn = self.ycolumn + "_yes"
        # Isolate X and y
        y = df[transformed_ycolumn]
        X = df.drop(columns=[transformed_ycolumn])
        return X, y

    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test


def save_drift_output(X_test_drifted, y_test, y_pred, y_pred_drifted, filename):
    test_results = X_test_drifted.copy()
    test_results['true_label'] = y_test.values
    test_results['y_pred'] = y_pred
    test_results['y_pred_drifted'] = y_pred_drifted
    test_results.head()
    test_results.to_csv(filename, index=False)

def save_normal_output(X_test, y_test, y_pred, filename):
    test_results = X_test.copy()
    test_results['true_label'] = y_test.values
    test_results['y_pred'] = y_pred
    test_results.head()
    test_results.to_csv(filename, index=False)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = "data/credit/CreditCard.csv"
    ycolumn = "card"
    data_prep = DataPreparation(filename, ycolumn)
    X, y = data_prep.load_dataset_file()
    X_train, X_test, y_train, y_test = data_prep.split_train_test(X, y)
    # Initialize the RandomForestDrift class
    rf_drift = RandomForestDrift(n_estimators=410, random_state=42, max_depth=4)
    # Fit the model
    rf_drift.fit(X_train, y_train)
    # Make predictions
    y_pred = rf_drift.predict(X_test)
    # Save the output
    # save_output(X_test, y_test, y_pred, None, "output/credit/normal_output.csv")
    save_normal_output(X_test, y_test, y_pred, "output/credit/normal_output.csv")
    # Show confusion matrix
    print("--------no drift--------")
    rf_drift.show_confusion_matrix(y_test, y_pred)

    # Make data quality drift predictions
    X_test_drift = rf_drift.data_drift_injection(X_test)
    y_pred_drift = rf_drift.predict(X_test_drift)
    X_test_quality = rf_drift.data_quality_injection(X_test)
    y_pred_quality = rf_drift.predict(X_test_quality)
    print("----------data quality--------")
    rf_drift.show_psi(y_pred, y_pred_quality)
    rf_drift.show_confusion_matrix(y_test, y_pred_quality)
    save_drift_output(X_test_drifted=X_test_drift,y_test=y_test,y_pred=y_pred,y_pred_drifted=y_pred_drift,filename="output/credit/data_drift_output.csv")
    print("----------data drift--------")
    rf_drift.show_psi(y_pred, y_pred_drift)
    rf_drift.show_confusion_matrix(y_test, y_pred_drift)
    save_drift_output(X_test_drifted=X_test_quality,y_test=y_test,y_pred=y_pred,y_pred_drifted=y_pred_quality,filename="output/credit/data_quality_output.csv")


