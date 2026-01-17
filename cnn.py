import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader

try:
    import cv2
except ImportError:
    cv2 = None


# --- Model Architecture ---
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
    
    
# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CACHED_MODEL = None

# Hyperparamenters
IN_CHANNELS = 1
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Load data
def load_mnist_data(batch_size=BATCH_SIZE):
    train_dataset = datasets.MNIST( root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Initialize network
def build_model(device=DEVICE):
    return CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)


# Train Network
def train_network(model, train_loader, device=DEVICE, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(num_epochs):
        for data, targets in train_loader:
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def load_or_train_model(model_path="mnist_cnn.pth", device=DEVICE):
    model = build_model(device=device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    train_loader, test_loader = load_mnist_data()
    train_network(model, train_loader, device=device)
    torch.save(model.state_dict(), model_path)

    train_acc = check_accuracy(train_loader, model, device=device)
    test_acc = check_accuracy(test_loader, model, device=device)
    print(f"Accuracy on training set: {train_acc * 100:.2f}%")
    print(f"Accuracy on test set: {test_acc * 100:.2f}%")

    model.eval()
    return model

def get_model(model_path="mnist_cnn.pth", device=DEVICE):
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = load_or_train_model(model_path=model_path, device=device)
    return _CACHED_MODEL

# --- Image Processing & Prediction ---
def _resize_with_padding(gray_image, size=28):
    if cv2 is None:
        raise RuntimeError("cv2 is required for number region preprocessing")

    h, w = gray_image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("number_region must be a non-empty image array")

    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim), dtype=gray_image.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    padded[y_offset : y_offset + h, x_offset : x_offset + w] = gray_image

    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_number_region(number_region):
    if cv2 is None:
        raise RuntimeError("cv2 is required for number region preprocessing")

    if number_region is None:
        raise ValueError("number_region cannot be None")

    if number_region.ndim == 3:
        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = number_region

    resized = _resize_with_padding(gray, size=28)
    normalized = resized.astype(np.float32) / 255.0

    # MNIST is white-on-black; invert if the input is black-on-white
    if normalized.mean() > 0.5:
        normalized = 1.0 - normalized

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_digit_with_confidence(number_region, model, device=DEVICE):
    model.eval()
    tensor = preprocess_number_region(number_region).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    return int(prediction.item()), float(confidence.item())

def _extract_digit_regions(number_region):
    if cv2 is None:
        raise RuntimeError("cv2 is required for digit extraction")

    if number_region.ndim == 3:
        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = number_region

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = 255 - binary

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 20: # Noise filter
            continue
        boxes.append((x, y, w, h))

    # Sort boxes from left to right
    boxes.sort(key=lambda b: b[0])
    digit_regions = [gray[y : y + h, x : x + w] for (x, y, w, h) in boxes]
    return digit_regions

def predict_number_from_region(number_region, model, device=DEVICE):
    value, _ = predict_number_with_confidence(number_region, model, device=device)
    return value

def predict_number_with_confidence(number_region, model, device=DEVICE):
    digit_regions = _extract_digit_regions(number_region)
    if not digit_regions:
        return 0, 0.0

    digits = []
    confidences = []
    for region in digit_regions:
        digit, confidence = predict_digit_with_confidence(region, model, device=device)
        digits.append(str(digit))
    
        confidences.append(confidence)

    try:
        value = int("".join(digits))
    except ValueError:
        return 0, 0.0

    if not (1 <= value <= 15):
        return 0, min(confidences) if confidences else 0.0

    return value, min(confidences) if confidences else 0.0

# --- Entry Point ---
if __name__ == "__main__":
    # This will train the model once and save it to 'mnist_cnn.pth'
    trained_model = load_or_train_model()
    print("Model ready for predictions.")
