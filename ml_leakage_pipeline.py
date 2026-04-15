# Task 1 — Reproduce and Identify Leakage Using the synthetic dataset generated below, scale the features on the entire dataset before splitting, 
# train a Logistic Regression model, and report train and test accuracy. Identify what is wrong with this approach.

# Task 1 Code:

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))


# Task1 Mistake 

# Data LEAKAGE: Here the scaling is done before the data split. Hence the scaler sees full dataset including test rows. The correct order is below.
# Split first → then fit the scaler on training data → transform both train and test







# Task 2 — Fix the Workflow Using a Pipeline Refactor the code from Task 1 using a Pipeline that combines StandardScaler and LogisticRegression. 
# Split the data first using train_test_split, then run 5-fold cross-validation using cross_val_score. Report mean accuracy and standard deviation.


# Task2 Code:

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split first — test set is now fully isolated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("CV Scores:", scores)
print("Mean Accuracy:", round(np.mean(scores), 4))
print("Std Deviation:", round(np.std(scores), 4))








# Task 3 — Experiment with Decision Tree Depth Train a DecisionTreeClassifier at max_depth values of 1, 5, and 20 using the same train-test split from Task 2. 
# Record train and test accuracy for each depth in a table and briefly explain which depth best balances fit and generalization.


# Task3 Code:

from sklearn.tree import DecisionTreeClassifier

depths = [1, 5, 20]

print(f"{'Depth':<10} {'Train Acc':<15} {'Test Acc'}")
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = round(dt.score(X_train, y_train), 4)
    test_acc = round(dt.score(X_test, y_test), 4)
    print(f"{d:<10} {train_acc:<15} {test_acc}")
