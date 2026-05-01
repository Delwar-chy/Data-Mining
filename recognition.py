from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
# Load dataset
digits = load_digits()

# Show one sample image
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Label: {digits.target[0]}")
plt.show()

# Data
X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, prediction)
print("\nConfusion Matrix:\n", cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Recall Score (macro = average recall for all digits)
recall = recall_score(y_test, prediction, average='macro')
print("Recall (Macro):", recall)
# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Show prediction example
plt.imshow(X_test[0].reshape(8,8), cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.show()