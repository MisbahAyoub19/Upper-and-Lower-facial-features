import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Dense, LeakyReLU
import seaborn as sns
import matplotlib.pyplot as plt

# Load features and labels from CSV files
features_df = pd.read_csv("last_2_features_oulu-vis_5.csv")
labels_df = pd.read_csv("combine-oulu-vi5-annotation.csv")

# Define label reference dictionary
points_dic = {'Anger': 0, 'Disgust': 1,
'Fear': 2, 'Happy': 3, 'Surprise': 4,
 'Sadness': 5}

# points_dic = {'Anger': 0, 'Contempt': 1, 'Disgust': 2,
#  'Embarrass': 3, 'Fear': 4, 'Joy': 5, 'Neutral': 6,
#  'Pride': 7, 'Sadness': 8, 'Surprise': 9}
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the DNN model
# model = Sequential([
#     # Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
#     # #Dropout(0.5),
#     # Dense(256, activation='relu'),
#     # #Dropout(0.5),
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#    # Dropout(0.5),
#     Dense(64, activation='relu'),
#
#    # Dropout(0.5),
#     Dense(32, activation='relu'),
#     #Dropout(0.5),
#     Dense(len(points_dic), activation='softmax')
# ])

model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.1),  # Adding LeakyReLU activation
    Dense(64),
    LeakyReLU(alpha=0.1),  # Adding LeakyReLU activation
    Dense(32),
    LeakyReLU(alpha=0.1),  # Adding LeakyReLU activation
    Dense(len(points_dic), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Track time
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Calculate overall processing time
processing_time = time.time() - start_time

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

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

# Save epoch details to CSV
epoch_details_df = pd.DataFrame(history.history)
epoch_details_df.to_csv("epoch_details.csv", index=False)

# Save accuracy details to CSV
accuracy_details_df = pd.DataFrame({
    "Test Loss": [loss],
    "Test Accuracy": [accuracy],
    "Processing Time (s)": [processing_time]
})
accuracy_details_df.to_csv("accuracy_details.csv", index=False)

# Plot confusion matrix with emotion labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=points_dic.keys(), yticklabels=points_dic.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Print accuracy per emotion
for emotion, accuracy in zip(points_dic.keys(), accuracy_per_emotion):
    print(f"Accuracy for {emotion}: {accuracy}")