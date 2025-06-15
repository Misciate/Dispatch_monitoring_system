import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ==== CẤU HÌNH ====
data_dir = 'Dataset/classification_flat'
batch_size = 32
num_epochs = 25  
learning_rate = 1e-4
num_classes = 6  # 6 lớp: dish/empty, dish/not_empty, dish/kakigori, tray/empty, tray/not_empty, tray/kakigori

# ==== TIỀN XỬ LÝ ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== LOAD DATA (định nghĩa ở mức module) ===
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ==== CHỈ CHẠY HUẤN LUYỆN KHI LÀ MAIN ===
if __name__ == '__main__':
    # ==== IN RA TÊN CÁC LỚP (định nghĩa ở mức module) ===
    print("Class mapping:")
    for idx, class_name in enumerate(dataset.classes):
        print(f"{idx}: {class_name}")
    print("Total classes:", len(dataset.classes))

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
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mobilenet_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        mobilenet_model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = mobilenet_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('Models', exist_ok=True)
            torch.save(mobilenet_model.state_dict(), 'Models/mobilenetv2_classification.pth')
            print("✅ Saved best model to Models/mobilenetv2_classification.pth")

    mobilenet_model.train()

    print("✅ Training completed!")