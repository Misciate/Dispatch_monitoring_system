import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== CẤU HÌNH ====
data_dir = 'Dataset/classification_flat'  # Thư mục chứa các ảnh phân loại theo 6 lớp
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
num_classes = 6  # dish/empty, dish/not_empty, dish/kakigori, tray/empty, tray/not_empty, tray/kakigori

# ==== TIỀN XỬ LÝ ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== LOAD DATA ====
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ==== IN RA TÊN CÁC LỚP ====
print("Class mapping:")
for idx, class_name in enumerate(dataset.classes):
    print(f"{idx}: {class_name}")

# ==== LOAD MODEL ====
mobilenet_model = models.mobilenet_v2(weights='DEFAULT')
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, num_classes)

# ==== SỬ DỤNG GPU (nếu có) ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilenet_model = mobilenet_model.to(device)

# ==== HUẤN LUYỆN ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilenet_model.parameters(), lr=learning_rate)

mobilenet_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = mobilenet_model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

# ==== LƯU MODEL ====
os.makedirs('Models', exist_ok=True)
torch.save(mobilenet_model.state_dict(), 'Models/mobilenetv2_classification.pth')
print("✅ Saved model to Models/mobilenetv2_classification.pth")
