import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# MNIST is the dataset used for digit recognition
# 70,000 images of handwritten digits (0–9)

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

print(f"Dataset shape: {X.shape}")   
print(f"Labels shape:  {y.shape}")   

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# trains the model ─
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#

# 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     # Never fit on test data!

# ── Step 6: Train Logistic Regression ────────────────────────
lr = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n── Logistic Regression ──")
print(classification_report(y_test, y_pred_lr))

# ── Step 7: Train SVM (better accuracy) ──────────────────────
svm = SVC(kernel='rbf', C=5, gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\n── SVM (RBF kernel) ──")
print(classification_report(y_test, y_pred_svm))

# ── Step 8: Confusion Matrix ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, model, name, preds in zip(
    axes,
    [lr, svm],
    ["Logistic Regression", "SVM"],
    [y_pred_lr, y_pred_svm]
):
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# ── Step 9: Predict your own drawing ─────────────────────────
from PIL import Image

def predict_image(path, model, scaler):
    img = Image.open(path).convert('L').resize((28, 28))
    pixels = np.array(img).flatten().reshape(1, -1)
    pixels_scaled = scaler.transform(pixels)
    pred = model.predict(pixels_scaled)[0]
    proba = model.predict_proba(pixels_scaled)[0] if hasattr(model, 'predict_proba') else None
    print(f"Predicted digit: {pred}")
    if proba is not None:
        print(f"Confidence: {proba[pred]*100:.1f}%")

# predict_image("my_digit.png", lr, scaler)
