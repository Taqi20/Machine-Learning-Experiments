import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_svm_experiment(random_state: int = 42):
    # 1. Load the dataset
    print("Step 1: Loading the Breast Cancer dataset...")
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    print("Dataset loaded successfully.")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print("-" * 40)

    # 2. Split the data into training and testing sets (70% train, 30% test)
    print("Step 2: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    print("Data split complete (70% training, 30% testing).")
    print("-" * 40)

    # 3. Scale the features
    print("Step 3: Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling complete.")
    print("-" * 40)

    # --- Implementation of Linear SVM ---
    print("--- Model 1: Linear SVM ---")
    print("Step 4: Training the Linear SVM model (kernel='linear')...")
    linear_svm = SVC(kernel="linear", random_state=random_state)
    linear_svm.fit(X_train_scaled, y_train)
    print("Linear SVM model training complete.")

    # 5. Evaluate the Linear SVM
    print("\nStep 5: Evaluating the Linear SVM model...")
    y_pred_linear = linear_svm.predict(X_test_scaled)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Accuracy (Linear SVM): {accuracy_linear:.4f}")
    print("\nClassification Report (Linear SVM):")
    print(classification_report(y_test, y_pred_linear, target_names=target_names))
    print("Confusion Matrix (Linear SVM):")
    print(confusion_matrix(y_test, y_pred_linear))
    print("-" * 40)

    # --- Implementation of Non-Linear SVM (RBF) ---
    print("\n--- Model 2: Non-Linear SVM (RBF Kernel) ---")
    print("Step 6: Training the Non-Linear SVM model (kernel='rbf')...")
    rbf_svm = SVC(kernel="rbf", random_state=random_state)
    rbf_svm.fit(X_train_scaled, y_train)
    print("Non-Linear SVM model training complete.")

    # 7. Evaluate the RBF SVM
    print("\nStep 7: Evaluating the Non-Linear SVM model...")
    y_pred_rbf = rbf_svm.predict(X_test_scaled)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"Accuracy (RBF SVM): {accuracy_rbf:.4f}")
    print("\nClassification Report (RBF SVM):")
    print(classification_report(y_test, y_pred_rbf, target_names=target_names))
    print("Confusion Matrix (RBF SVM):")
    print(confusion_matrix(y_test, y_pred_rbf))
    print("-" * 40)

    # 8. Visualize the Decision Boundary using only the first two features
    print("\nStep 8: Visualizing the SVM Decision Boundary (first two features)...")
    X_viz = X[:, :2]
    y_viz = y

    scaler_viz = StandardScaler()
    X_viz_scaled = scaler_viz.fit_transform(X_viz)

    svm_viz = SVC(kernel="linear", random_state=random_state)
    svm_viz.fit(X_viz_scaled, y_viz)

    # create meshgrid
    h = 0.02  # step size
    x_min, x_max = X_viz_scaled[:, 0].min() - 1, X_viz_scaled[:, 0].max() + 1
    y_min, y_max = X_viz_scaled[:, 1].min() - 1, X_viz_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # predict for each point in the mesh
    Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_viz_scaled[:, 0], X_viz_scaled[:, 1], c=y_viz, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.xlabel(feature_names[0] + " (scaled)")
    plt.ylabel(feature_names[1] + " (scaled)")
    plt.title("SVM Decision Boundary (first two features)")
    plt.show()
    print("Visualization complete.")


if __name__ == "__main__":
    run_svm_experiment()
