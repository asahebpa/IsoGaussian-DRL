import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
import os
import csv
import math
import argparse

from utils.utils import (
    parse_mlp_depth, 
    parse_mlp_width, 
    get_optimizer
)

# ---------------------- SIGReg ----------------------
class SIGReg(nn.Module):
    def __init__(self, embedding_dim, num_slices=16, num_t=8, t_max=5.0, device="cuda"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_slices = num_slices
        self.num_t = num_t
        self.t_max = t_max
        self.device = device
        self.register_buffer("t_grid", torch.linspace(-t_max, t_max, steps=num_t))

    def forward(self, embeddings):
        B, D = embeddings.shape
        a = torch.randn(self.num_slices, D, device=embeddings.device)
        a = a / (a.norm(dim=1, keepdim=True) + 1e-12)
        s = torch.matmul(embeddings, a.t()).t()
        t = self.t_grid.to(embeddings.device)
        loss = 0.0
        for ti in range(self.num_t):
            tt = t[ti]
            cos_ts = torch.cos(tt * s)
            sin_ts = torch.sin(tt * s)
            re = cos_ts.mean(dim=1)
            im = sin_ts.mean(dim=1)
            target = math.exp(-0.5 * (tt.item() ** 2))
            loss += ((re - target) ** 2 + (im ** 2)).mean()
        return loss / float(self.num_t)

# ---------------------- NonStationaryDataset ----------------------
class NonStationaryDataset(Dataset):
    def __init__(self, dataset, permutation=None):
        self.dataset = dataset
        self.permutation = permutation if permutation is not None else list(range(len(dataset.targets)))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.dataset.targets[self.permutation[idx]]
        return image, label
    def reshuffle_labels(self):
        n = len(self.dataset.targets)
        self.permutation = np.random.permutation(n).tolist()

# ---------------------- MLP Helper ----------------------
def get_mlp(mlp_type, input_size, mlp_width, mlp_depth, use_ln=False, device="cuda"):
    if mlp_type == "default":
        from models.mlp import MLP as MLPClass
    elif mlp_type == "residual":
        from models.mlp import ResidualMLP as MLPClass
    elif mlp_type == "multiskip_residual":
        from models.mlp import MultiSkipResidualMLP as MLPClass
    else:
        raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
    
    trunk_hidden_size = parse_mlp_width(mlp_width)
    trunk_num_layers = parse_mlp_depth(mlp_depth, mlp_type)
    
    mlp = MLPClass(
        input_size=int(input_size),
        hidden_size=trunk_hidden_size,
        output_size=512,
        num_layers=trunk_num_layers,
        use_ln=use_ln,
        activation_fn="relu",
        device=device,
        last_act=True
    )
    return mlp

# ---------------------- CNN+MLP Model ----------------------
class CIFARClassifier(nn.Module):
    def __init__(self, mlp, num_classes, use_ln=False):
        super().__init__()
        layers = [
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.2),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.2)
        ]
        if use_ln:
            layers.insert(5, nn.GroupNorm(8,64))
            layers.insert(12, nn.GroupNorm(16,128))
            layers.insert(19, nn.GroupNorm(16,128))
        self.cnn = nn.Sequential(*layers)
        self.network = nn.Sequential(self.cnn, nn.Flatten())
        self.trunk = mlp
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        features = self.network(x)
        trunk_features = self.trunk(features)
        outputs = self.classifier(trunk_features)
        return outputs, trunk_features

# ---------------------- Training ----------------------
def train_cifar(run_name, mlp_type, optimizer_name, mlp_depth, mlp_width, dataset_name="cifar100",
                non_stationary=False, epochs=25, batch_size=128, use_ln=False, device="cuda", lambda_sig=1.0):
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    wandb.init(
        project=f"{dataset_name}_good",
        name=run_name,
        config={
            "mlp_type": mlp_type,
            "optimizer": optimizer_name,
            "mlp_depth": mlp_depth,
            "mlp_width": mlp_width,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.00025,
            "dataset": dataset_name,
            "non_stationary": non_stationary,
            "use_ln": use_ln,
            "lambda_sig": lambda_sig
        }
    )
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset_class = torchvision.datasets.CIFAR100 if dataset_name=="cifar100" else torchvision.datasets.CIFAR10
    num_classes = 100 if dataset_name=="cifar100" else 10
    num_datapoints = 10000
    
    trainset_base = dataset_class(root=f'./{dataset_name}_data', train=True, download=True, transform=transform)
    testset_base = dataset_class(root=f'./{dataset_name}_data', train=False, download=True, transform=transform)
    train_indices = list(range(num_datapoints))
    test_indices = list(range(num_datapoints))
    trainset_base.targets = [trainset_base.targets[i] for i in train_indices]
    testset_base.targets = [testset_base.targets[i] for i in test_indices]
    if hasattr(trainset_base, 'data'):
        trainset_base.data = trainset_base.data[train_indices]
        testset_base.data = testset_base.data[test_indices]
    else:
        trainset_base.imgs = [trainset_base.imgs[i] for i in train_indices]
        testset_base.imgs = [testset_base.imgs[i] for i in test_indices]
    
    trainset = NonStationaryDataset(trainset_base) if non_stationary else trainset_base
    testset = NonStationaryDataset(testset_base) if non_stationary else testset_base
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    mlp_input_size = 2048
    mlp = get_mlp(mlp_type, input_size=mlp_input_size, mlp_width=mlp_width, mlp_depth=mlp_depth, use_ln=use_ln, device=device)
    model = CIFARClassifier(mlp, num_classes, use_ln=use_ln).to(device)
    sigreg = SIGReg(embedding_dim=512, num_slices=16, num_t=8, t_max=5.0, device=device).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name)(model.parameters(), lr=0.00025)
    
    for epoch in range(epochs):
        if non_stationary and epoch in [20,40,60,80]:
            trainset.reshuffle_labels()
            testset.reshuffle_labels()
            print(f"Reshuffling labels at epoch {epoch}")
        
        model.train()
        total_loss, correct, total, total_sigreg = 0,0,0,0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, trunk_features = model(images)
            loss_ce = criterion(outputs, labels)
            loss_sig = sigreg(trunk_features)
            loss = loss_ce  + (lambda_sig * loss_sig)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sigreg += loss_sig.item() 
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss = total_loss / len(trainloader)
        train_sigreg_loss = total_sigreg / len(trainloader)
        train_acc = correct / total
        
        model.eval()
        total_loss, correct, total, total_sigreg = 0,0,0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs, trunk_features = model(images)
                loss_ce = criterion(outputs, labels)
                loss_sig = sigreg(trunk_features)
                loss = loss_ce + (lambda_sig * loss_sig)
                total_loss += loss.item()
                total_sigreg += loss_sig.item() 
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        test_loss = total_loss / len(testloader)
        test_sigreg_loss = total_sigreg / len(testloader)
        test_acc = correct / total
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "sigreg_loss": loss_sig.item(),
            "total test sigreg loss": test_sigreg_loss,
            "total train sigreg loss": train_sigreg_loss,
        })
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, SIGReg Loss={loss_sig.item():.4f}")
    
    wandb.finish()

# ---------------------- Main ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_type", type=str, required=True)
    parser.add_argument("--mlp_depth", type=str, required=True)
    parser.add_argument("--mlp_width", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--non_stationary", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--use_ln", action="store_true")
    parser.add_argument("--lambda_sig", type=float, default=1.0)
    args = parser.parse_args()
    
    run_name = f"sigreg_DATA:{args.dataset}_MLP.TYPE:{args.mlp_type}_MLP.DEPTH:{args.mlp_depth}_MLP.WIDTH:{args.mlp_width}_OPTIM:{args.optimizer}_NS:{args.non_stationary}_LN:{args.use_ln}"
    
    train_cifar(
        run_name,
        args.mlp_type,
        args.optimizer,
        args.mlp_depth,
        args.mlp_width,
        dataset_name=args.dataset,
        non_stationary=args.non_stationary, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        use_ln=args.use_ln,
        lambda_sig=args.lambda_sig
    )
