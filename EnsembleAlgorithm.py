# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  # Used as base learner for AdaBoost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_ensemble_experiment():
    """
    Runs both Bagging and Boosting experiments using the Wine dataset.
    """

    # 1. Load the dataset
    print("Step 1: Loading the Wine dataset...")
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    print("Dataset loaded successfully.")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of classes: {len(target_names)}")
    print("-" * 50)

    # 2. Split the data into training and testing sets
    print("Step 2: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Data split complete (70% training, 30% testing).")
    print("-" * 50)

    # 3. Scale the features
    # Although tree-based models are less sensitive to feature scaling, it’s still good practice.
    print("Step 3: Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling complete.")
    print("-" * 50)

    # --- Implementation of Bagging: Random Forest ---
    print("--- Model 1: Bagging (RandomForestClassifier) ---")
    print("Step 4: Training the RandomForest model...")
    bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)
    bagging_model.fit(X_train_scaled, y_train)
    print("RandomForest model training complete.")

    # 5. Evaluate the Bagging model
    print("\nStep 5: Evaluating the RandomForest model...")
    y_pred_bagging = bagging_model.predict(X_test_scaled)
    accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
    print(f"Accuracy: {accuracy_bagging:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_bagging, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_bagging))
    print("-" * 50)

    # --- Implementation of Boosting: AdaBoost ---
    print("\n--- Model 2: Boosting (AdaBoostClassifier) ---")
    print("Step 6: Training the AdaBoost model...")

    # Base learner: shallow decision tree (decision stump)
    weak_learner = DecisionTreeClassifier(max_depth=1)

    boosting_model = AdaBoostClassifier(
        estimator=weak_learner, n_estimators=100, random_state=42
    )
    boosting_model.fit(X_train_scaled, y_train)
    print("AdaBoost model training complete.")

    # 7. Evaluate the Boosting model
    print("\nStep 7: Evaluating the AdaBoost model...")
    y_pred_boosting = boosting_model.predict(X_test_scaled)
    accuracy_boosting = accuracy_score(y_test, y_pred_boosting)
    print(f"Accuracy: {accuracy_boosting:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_boosting, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_boosting))
    print("-" * 50)

    # --- Final Comparison ---
    print("\n--- Final Performance Comparison ---")
    print(f"Random Forest (Bagging) Accuracy: {accuracy_bagging:.4f}")
    print(f"AdaBoost (Boosting) Accuracy: {accuracy_boosting:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    run_ensemble_experiment()
