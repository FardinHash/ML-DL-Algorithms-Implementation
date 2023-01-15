# import libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis = 1), data["target"], test_size = 0.3)

# Initialize the classifier
gnb = GaussianNB()

# Train the classifier on the train data
gnb.fit(X_train, y_train)

# Test the classifier on the test data
accuracy = gnb.score(X_test, y_test)
print("Accuracy:", accuracy)
