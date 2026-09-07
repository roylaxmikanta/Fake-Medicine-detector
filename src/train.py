import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    IMG_SIZE = 150
    BATCH_SIZE = 32
    EPOCHS = 5  # Keep it short for simulation purposes

    # Data augmentations
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names = full_dataset.classes
    print(f"Class names: {class_names}")

    # Split dataset 80% train+val, 20% test
    total_len = len(full_dataset)
    test_len = int(0.2 * total_len)
    train_val_len = total_len - test_len
    
    generator = torch.Generator().manual_seed(42)
    train_val_dataset, test_dataset = random_split(full_dataset, [train_val_len, test_len], generator=generator)

    # Split train+val into 80% train, 20% val
    val_len = int(0.2 * train_val_len)
    train_len = train_val_len - val_len
    train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len], generator=generator)

    # Apply transforms using a wrapper dataset
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)

    train_data = TransformedDataset(train_dataset, transform=train_transforms)
    val_data = TransformedDataset(val_dataset, transform=test_transforms)
    test_data = TransformedDataset(test_dataset, transform=test_transforms)

    print(f"Sizes -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Build model (MobileNetV2)
    # Use standard models instead of 'weights' argument if backwards compatibility is needed
    model = models.mobilenet_v2(pretrained=True)
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print("Training model...")
    best_val_acc = 0.0
    model_path = os.path.join(MODEL_DIR, 'medicine_model.pt')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix on Test Set')
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
    
    mapping = {'class_names': class_names}
    with open(os.path.join(MODEL_DIR, 'preprocessing.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
        
    print(f"Model saved to {model_path} and preprocessing mapping to preprocessing.pkl")

if __name__ == '__main__':
    main()