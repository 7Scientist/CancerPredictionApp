# 1. Data Preprocessing & Exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("C:/Users/DAVIS MYRE/Desktop/NOTEPADS/cancer patient data sets.csv")

# Check for missing values
print(df.isnull().sum())

# Handle missing values

# Impute categorical missing values with most frequent value
imputer_cat = SimpleImputer(strategy="most_frequent")
df[df.select_dtypes(include=["object"]).columns] = imputer_cat.fit_transform(df.select_dtypes(include=["object"]))





# EDA - Summary statistics
print(df.describe())

# Visualizing distributions
df.hist(figsize=(12, 8))
# plt.show()

# Handle outliers
from scipy.stats import zscore

z_scores = np.abs(zscore(df.select_dtypes(include=["number"])))

df = df[(z_scores < 3).all(axis=1)]

# Correlation heatmap
# Drop non-numeric columns before computing correlation
df_numeric = df.select_dtypes(include=["number"])

# Computation and plottting the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()





# Encoding categorical variables using Label Encoding
encoder = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = encoder.fit_transform(df[col])




# 2. Feature Engineering
# Define target column
target_col = "Level"  # This is the column to predict

# Compute correlation matrix
correlation = df.corr()

# Select highly correlated features
high_correlation_features = correlation.index[abs(correlation[target_col]) > 0.3]

# Keep only selected features
df_selected = df[high_correlation_features]

# Display selected features
print("Selected Features:\n", df_selected.columns)






# Split features and target variable
X = df.drop(columns=["Level"])  # Features
y = df["Level"]  # Target variable

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing complete!")






# Model training
# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Training and evaluating the models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 50)




# Hyperparameter tuning
# 1. Logistic Regression
param_grid_lr = {
    "C": [0.01, 0.1, 1, 10],  # Regularization strength
    "solver": ["liblinear", "lbfgs"]
}
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_lr.fit(X_train, y_train)

# 2. Decision Tree
param_grid_dt = {
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

# 3. Random Forest
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Print best parameters & accuracy for each model
print("Best Logistic Regression Parameters:", grid_search_lr.best_params_)
print("Best Logistic Regression Accuracy:", grid_search_lr.best_score_)

print("Best Decision Tree Parameters:", grid_search_dt.best_params_)
print("Best Decision Tree Accuracy:", grid_search_dt.best_score_)

print("Best Random Forest Parameters:", grid_search_rf.best_params_)
print("Best Random Forest Accuracy:", grid_search_rf.best_score_)





# Make predictions
y_pred_lr = grid_search_lr.best_estimator_.predict(X_test)
y_pred_dt = grid_search_dt.best_estimator_.predict(X_test)
y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)

# Function to print evaluation metrics
def evaluate_model(model_name, y_test, y_pred):
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1-score:", f1_score(y_test, y_pred, average="weighted"))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Evaluate each model
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)





from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_regression(model_name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Regression Performance:")
    print("RMSE:", rmse)
    print("R-squared:", r2)
evaluate_regression("Random Forest Regressor", y_test, y_pred_rf)







# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Plot confusion matrix for each model
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")





from sklearn.model_selection import cross_val_score

# Perform cross-validation (10-fold)
models = {
    "Logistic Regression": grid_search_lr.best_estimator_,
    "Decision Tree": grid_search_dt.best_estimator_,
    "Random Forest": grid_search_rf.best_estimator_
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    print(f"{name} - Mean Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")






# Interpreting feature importance for tree-based models.
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)  # Convert to DataFrame

# Extract feature names
feature_names = X_train.columns

# Get feature importances from Random Forest
feature_importances = grid_search_rf.best_estimator_.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="coolwarm")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()





#### Limitations:

# If the model is overfitting, it won’t generalize well.

# If accuracy is too high, check for leakage or biases in the data.

#### Potential Improvements:

# Collect more diverse data to prevent overfitting.

# Use hyperparameter tuning to fine-tune performance.

# Apply regularization to reduce model complexity.

# Saving the created model as Mybest_model.pkl
model = RandomForestClassifier()
param_grid = {'n_estimators': [10, 50, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
joblib.dump(best_model, "Mybest_model.pkl")

