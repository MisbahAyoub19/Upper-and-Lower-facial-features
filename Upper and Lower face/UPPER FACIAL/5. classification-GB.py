import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load features and labels from CSV files
features_df = pd.read_csv("features/features-adfes.csv")
labels_df = pd.read_csv("annotations/combine-adfes-annotation.csv")

# Define label reference dictionary
#points_dic = {'Anger': 0, 'Contempt': 1, 'Disgust': 2,'Embarrass': 3, 'Fear': 4, 'Joy': 5, 'Neutral': 6,'Pride': 7, 'Sadness': 8, 'Surprise': 9}

#points_dic = {'Anger': 0, 'Disgust': 1,'Fear': 2, 'Happy': 3, 'Surprise': 4, 'Sadness': 5}
points_dic = {'Anger': 0, 'Contempt': 1, 'Disgust': 2,
 'Embarrass': 3, 'Fear': 4, 'Joy': 5, 'Neutral': 6,
 'Pride': 7, 'Sadness': 8, 'Surprise': 9}

# Select only the numeric columns from features_df
numeric_columns = features_df.select_dtypes(include=np.number).columns
X = features_df[numeric_columns].values

# Extract labels
y = labels_df.values.ravel()

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Inverse mapping for emotions
inv_points_dic = {v: k for k, v in points_dic.items()}

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Gradient Boosting model
#model = GradientBoostingClassifier()
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=5,
                                   min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42)
# Track time
start_time = time.time()

# Train the model
model.fit(X_train, y_train)

# Calculate overall processing time
processing_time = time.time() - start_time

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predictions
y_pred = model.predict(X_test)

# Calculate F1 score
f1_score = classification_report(y_test, y_pred)
print("F1 Score:")
print(f1_score)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy per emotion
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
accuracy_per_emotion = precision_recall_fscore_support(y_test, y_pred, average=None)[0]

# Plot confusion matrix with emotion labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Greens", fmt="d",
            xticklabels=points_dic.keys(), yticklabels=points_dic.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Convert confusion matrix to DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, columns=label_encoder.classes_, index=label_encoder.classes_)

# Save confusion matrix to CSV
conf_matrix_df.to_csv("confusion_matrix_adfes_GB.csv")

# Print accuracy per emotion
for emotion, accuracy in zip(points_dic.keys(), accuracy_per_emotion):
    print(f"Accuracy for {emotion}: {accuracy}")
