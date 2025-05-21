# train_PointNet2.py
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from multiprocessing import freeze_support
from pointnet2_utils import (
    PointNetSetAbstraction,
    farthest_point_sample,
    query_ball_point,
    index_points,
    sample_and_group_all
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- load data --------------------
class PointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=4096):
        self.num_points = num_points
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.file_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.txt')]
            self.file_paths.extend([os.path.join(cls_dir, f) for f in files])
            self.labels.extend([idx] * len(files))

        print(f"Finished initialize, find {len(self.classes)} classes，total sample：{len(self)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load txt point cloud
        try:
            points = np.loadtxt(self.file_paths[idx], delimiter=',', dtype=np.float32)[:self.num_points, :6]
        except ValueError:
            points = np.loadtxt(self.file_paths[idx], dtype=np.float32)[:self.num_points, :6]

        if points.shape[0] < self.num_points:
            pad_size = self.num_points - points.shape[0]
            points = np.concatenate([points, points[:pad_size]], axis=0)
        elif points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]

        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid
        max_dist = np.max(np.sqrt(np.sum(points[:, :3] ** 2, axis=1)))
        points[:, :3] /= (max_dist + 1e-6)  # 防止除以零

        if points.shape[1] >= 6:
            points[:, 3:6] /= 255.0

        return torch.tensor(points, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------- PointNet2 model --------------------
class PointNet2Cls(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # SA模块配置
        self.sa1 = PointNetSetAbstraction(
            npoint=2048,
            radius=0.2,
            nsample=64,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=512,
            radius=0.4,
            nsample=128,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, N, C = x.size()

        # [B, N, 6] => [B, 6, N]
        x = x.permute(0, 2, 1)

        # Set Abstraction
        xyz = x[:, :3, :]
        features = x[:, 3:, :] if C > 3 else None

        l1_xyz, l1_features = self.sa1(xyz, features)
        # print('After sa1 processing: ','l1_xyz.shape',l1_xyz.shape,'l1_features.shape', l1_features.shape)

        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        # print('After sa2 processing: ', 'l2_xyz.shape', l2_xyz.shape, 'l2_features.shape', l2_features.shape)

        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        # print('After sa3 processing: ', 'l3_xyz.shape', l3_xyz.shape, 'l3_features.shape', l3_features.shape)

        global_features = l3_features.view(B, -1)

        return self.classifier(global_features)



def main():
    freeze_support()

    dataset = PointCloudDataset(r"D:\MASTERS\PGCAP__master\data\ABA_SHIFT_CLASSIFY")

    _, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        stratify=dataset.labels,
        random_state=42
    )

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,  #
        num_workers=2
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    # initialization
    model = PointNet2Cls(num_classes=len(dataset.classes)).to(device)
    test_input = torch.randn(2, 4096, 6).to(device)  # Foward test
    with torch.no_grad():
        test_output = model(test_input)
        print(f"Output shape：{test_output.shape} ")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    peak_loss=5000
    print("\n---------- Start training ----------")
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # evaluation
        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for points, labels in test_loader:
                points = points.to(device)
                labels = labels.to(device)

                outputs = model(points)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                correct += outputs.argmax(1).eq(labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        scheduler.step(avg_test_loss)

        avg_train_loss = train_loss / len(train_loader)
        accuracy = correct / len(test_idx)

        save_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(save_dict, r"D:\MASTERS\PGCAP__master\model_pth\last_model.pth")

        if avg_test_loss<peak_loss:
            peak_loss=avg_test_loss
            torch.save(save_dict, r"D:\MASTERS\PGCAP__master\model_pth\best_model.pth")
            print(f"※ Saved model, best accuracy is：{accuracy:.2%}")

        print(f"Epoch {epoch + 1:03d} | "
              f"Training loss：{avg_train_loss:.4f} | "
              f"Test loss：{avg_test_loss:.4f} | "
              f"Test accuracy：{accuracy:.2%} | "
              f"Learning rate：{optimizer.param_groups[0]['lr']:.2e}")


if __name__ == '__main__':
    main()
