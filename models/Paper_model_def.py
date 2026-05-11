import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2Wrapper:
    def __init__(self, num_classes=6, learning_rate=0.001):
        """
        Implements EfficientNetV2-Small (Published 2021).
        We use pre-trained ImageNet weights to speed up convergence, 
        but replace the final classification head for our Intel dataset.
        """
        # Automatically use GPU in Colab if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing EfficientNetV2 on device: {self.device}")
        
        # 1. Load the 2021 Architecture with base weights
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        
        # 2. Replace the final classification layer (Classifier head)
        # EfficientNetV2's classifier is a Sequential block; the final Linear layer is at index 1
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        # 3. Define Framework-native Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, dataloader):
        """Runs one epoch of PyTorch training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Note: 'dataloader' should be a PyTorch DataLoader object 
        # yielding (images, labels) batches.
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass & Optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader):
        """Runs PyTorch validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc