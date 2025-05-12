import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_map = {
            'Heishui_group': 0,
            'Jiuzhaigou_group': 1,
            'Songpan_group': 2
        }
        self.file_list = self._load_files()

    def _load_files(self):
        files = []
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.txt'):
                    files.append((os.path.join(class_dir, file), label))
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        data = np.loadtxt(file_path, delimiter=',')  # (n_points, 6)

        # Random sample (2048)
        n_points = data.shape[0]
        if n_points >= 2048:
            indices = np.random.choice(n_points, 2048, replace=False)
        else:
            indices = np.random.choice(n_points, 2048, replace=True)

        sampled_data = data[indices]

        # 分离xyz和rgb
        xyz = sampled_data[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz -= centroid  # 中心化
        rgb = sampled_data[:, 3:] / 255.0  # Normalization

        # Concatenate to (6, 2048) tensor
        point_cloud = np.vstack([xyz.T, rgb.T])

        return torch.FloatTensor(point_cloud), torch.tensor(label, dtype=torch.long)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, use_bn=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_bn = use_bn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if use_bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features):
        """
        xyz: (B, 3, N)
        features: (B, C, N)
        return: new_xyz, new_features
        """
        if features is not None:
            combined_features = torch.cat([xyz, features], dim=1)
        else:
            combined_features = xyz

        combined_features = combined_features.unsqueeze(-1)  # (B, C, N, 1)

        for i, conv in enumerate(self.mlp_convs):
            if self.use_bn:
                bn = self.mlp_bns[i]
                combined_features = F.relu(bn(conv(combined_features)))
            else:
                combined_features = F.relu(conv(combined_features))

        new_features = combined_features.squeeze(-1)
        return xyz, new_features


class PointNet2Cls(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNet2Cls, self).__init__()

        # SA1: (xyz+rgb)
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=6,  # xyz+rgb
            mlp=[64, 64, 128],
            use_bn=False
        )

        # SA2: 128+3=131
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256]
        )

        # SA3: Global feature
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024]
        )

        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        B, _, N = x.shape

        # Initial feature
        xyz = x[:, :3, :]  # (B, 3, N)
        features = x[:, 3:, :] if x.size(1) > 3 else None

        # Extracting feature
        xyz, features = self.sa1(xyz, features)  # (B, 128, 512)
        xyz, features = self.sa2(xyz, features)  # (B, 256, 128)
        _, features = self.sa3(xyz, features)  # (B, 1024, 1)

        # Global feature
        x = torch.max(features, 2)[0]  # (B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Ensuring batch_size > 1
        if data.size(0) == 1:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()


    print(f'Train Epoch: {epoch} ')
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train set: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if data.size(0) == 1:
                continue

            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


def main():
    # Config
    data_dir = './data/ABA_SHIFT_CLASSIFY'
    model_dir = './model_pth'
    os.makedirs(model_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Loading data
    dataset = PointCloudDataset(data_dir)
    train_size = int(0.6 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    # Ensuring batch_size > 1
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initializing model
    model = PointNet2Cls(num_classes=3).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Training parameters
    epochs = 1000
    best_acc = 0.0
    min_loss=1000
    # Training epoch
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()

        # Saving best model
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print(f'Saved best model in epoch {epoch}')

    print(f'Training complete! Best test accuracy: {test_acc:.2f}%')
    torch.save(model.state_dict(), os.path.join(model_dir, 'last_model.pth'))
    print('Saved final model')


if __name__ == '__main__':
    main()
