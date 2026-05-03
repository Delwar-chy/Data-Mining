import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# MNIST is the dataset used for digit recognition
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

# trains the model 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     # Never fit on test data!

# ── Step 6: Train Logistic Regression ────────────────────────
lr = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n── Logistic Regression ──")
print(classification_report(y_test, y_pred_lr))

# ── Step 7: Train SVM ──────────────────────
svm = SVC(kernel='rbf', C=5, gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\n── SVM (RBF kernel) ──")
print(classification_report(y_test, y_pred_svm))


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

# ── Step 9: Predict drawing ─────────────────────────
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



from PIL import Image

# ── Load MNIST ──────────────────────────────────────────────
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print("Done! Dataset shape:", X.shape)

# ── Using 10,000 samples ────────────────
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=10000, random_state=42, stratify=y
)

# ── Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

# ── Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train
print("Training model... (should take 20–40 seconds)")
model = LogisticRegression(
    max_iter=100,       
    solver='lbfgs',    
    n_jobs=1,         
    multi_class='auto'
)
model.fit(X_train, y_train)
print(f"Model accuracy: {model.score(X_test, y_test)*100:.1f}%")

# ── Predict your MS Paint image ─────────────────────────────
def predict_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    pixels = np.array(img)

    plt.figure(figsize=(3, 3))
    plt.imshow(pixels, cmap='gray')
    plt.title("What the model sees (28×28)")
    plt.axis('off')
    plt.show()

    pixels_flat = pixels.flatten().reshape(1, -1).astype(float)
    pixels_scaled = scaler.transform(pixels_flat)
    pred = model.predict(pixels_scaled)[0]

    proba = model.predict_proba(pixels_scaled)[0]
    print(f"\n Predicted digit : {pred}")
    print(f" Confidence      : {proba[pred]*100:.1f}%")
    print(f"\n All probabilities:")
    for digit, prob in enumerate(proba):
        bar = '█' * int(prob * 40)
        print(f"  {digit} | {bar:<40} {prob*100:.1f}%")

predict_image(r"C:\usegit\Data-Mining\my_digit.png")
predict_image(r"C:\usegit\Data-Mining\my_digit2.png")
#any heavy model my laptop cannot handle 
