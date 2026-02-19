import pytesseract
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import pytesseract
from src.feature_extraction import extract_features
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

dataset_path = "dataset"


classes = os.listdir(dataset_path)
print("Classes:", classes)

sample_class = classes[0]
sample_image_path = os.path.join(dataset_path, sample_class, 
                                os.listdir(os.path.join(dataset_path, sample_class))[0])

img = cv2.imread(sample_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Image Shape:", img.shape)

plt.imshow(img)
plt.title(sample_class)
plt.show()

transform = transforms.Compose([
    transforms.ToTensor()
])

tensor_img = transform(img)
print("Tensor Shape:", tensor_img.shape)


text = pytesseract.image_to_string(img)
print("Extracted Text:\n", text[:500])

features = extract_features(text)

print("\nExtracted Features:")
print("Word Count:", features[0])
print("Char Count:", features[1])
print("Avg Word Length:", features[2])
print("Digit Count:", features[3])
print("Uppercase Ratio:", features[4])
print("Number of Lines:", features[5])

all_features = []
labels = []

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        text = pytesseract.image_to_string(img)
        feature_vector = extract_features(text)
        
        all_features.append(feature_vector)
        labels.append(class_name)

all_features = np.array(all_features)

print("\nDataset Feature Matrix Shape:", all_features.shape)

mean_vector = np.mean(all_features, axis=0)
variance_vector = np.var(all_features, axis=0)
print("\nMean of Features:\n", mean_vector)
print("\nVariance of Features:\n", variance_vector)

cov_matrix = np.cov(all_features, rowvar=False)
print("\nCovariance Matrix:\n", cov_matrix)

# Mean Centering
centered_data = all_features - mean_vector
print("\nCentered Data Shape:", centered_data.shape)
cov_matrix_pca = np.cov(centered_data, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_pca)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print("\nSorted Eigenvalues:\n", eigenvalues)

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nExplained Variance Ratio:\n", explained_variance_ratio)

k = 2
principal_components = eigenvectors[:, :k]
reduced_data = np.dot(centered_data, principal_components)
print("\nReduced Data Shape:", reduced_data.shape)

plt.figure(figsize=(8,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
print("\nEncoded Classes:", label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    all_features, encoded_labels,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(all_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, encoded_labels,
    test_size=0.2,
    random_state=42
)

clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy (Scaled Features):",
    accuracy_score(y_test, y_pred))

print("\nClassification Report:\n",
    classification_report(y_test, y_pred))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_features)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(encoded_labels, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)
    
    accuracy = (predicted == y_tensor).float().mean()
    
print("\nNeural Network Accuracy:", accuracy.item())

torch.save(model.state_dict(), "models/nn_model.pth")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")