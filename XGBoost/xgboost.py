# import libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis = 1), data["target"], test_size = 0.3)

# Convert data into DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# hyperparameters for the model
params = {'objective': 'binary:logistic', 
          'max_depth': 2, 
          'learning_rate': 1.0, 
          'silent': 1.0, 
          'n_estimators': 5}

# Train the model
bst = xgb.train(params, dtrain)

# predictions
preds = bst.predict(dtest)

# Convert the predictions into binary labels
labels = (preds > 0.5).astype(int)

# Calculate the accuracy
accuracy = (labels == y_test).sum() / len(y_test)
print("Accuracy:", accuracy)
