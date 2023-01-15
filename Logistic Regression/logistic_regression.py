# import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.3)

# Initialize the classifier
clf = LogisticRegression()

# Train the classifier on the train data
clf.fit(X_train, y_train)

# Test the classifier on the test data
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict the label of a new sample
new_sample = [3, 6, 5, 4]
prediction = clf.predict([new_sample])
print("Prediction:", prediction)

