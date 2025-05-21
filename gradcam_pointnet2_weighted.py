import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
from pointnet2_utils import farthest_point_sample, query_ball_point, index_points


muti_weights=[0.65, 0.25, 0.1]

class PointNet2ClsWithHooks(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=2048, radius=0.2, nsample=64,
            in_channel=6, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=512, radius=0.4, nsample=128,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
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

        # save features of hooks
        self.features = {}
        self.gradients = {}
        print("Finished initialization，SA config:")
        print(f"- SA1: 2048, radius 0.2, 64 samples")
        print(f"- SA2: 512, radius 0.4, 128 samples")
        print(f"- SA3: global pooling")

    @staticmethod
    def amplify_gradient(grad):
        sign = torch.sign(grad)
        magnitude = torch.log1p(torch.abs(grad))
        return sign * magnitude

    def activations_hook(self, grad, name):
        raw_mean = grad.mean().item()
        raw_max = grad.max().item()

        amplified_grad = self.amplify_gradient(grad)
        self.gradients[name] = amplified_grad

        print(f"[Gradient hook] {name}")
        print(f"Original maximum value: {raw_max:.6f}, Amplified maximum value: {amplified_grad.max().item():.6f}")
        print(f"None zero propotion: {(grad != 0).float().mean().item():.2%}")

    def forward(self, x):
        B, N, C = x.size()
        print(f"\nInput shape: {x.shape}")

        x = x.permute(0, 2, 1)
        xyz = x[:, :3, :]
        features = x[:, 3:, :] if C > 3 else None

        # SA1
        l1_xyz, l1_features = self.sa1(xyz, features)
        self.features['sa1'] = l1_features
        self.features['sa1_xyz'] = l1_xyz
        if l1_features.requires_grad:
            h = l1_features.register_hook(lambda grad: self.activations_hook(grad, 'sa1'))
        # print(f"SA1 - coords: {l1_xyz.shape}, features: {l1_features.shape}")

        # SA2
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        self.features['sa2'] = l2_features
        self.features['sa2_xyz'] = l2_xyz
        if l2_features.requires_grad:
            h = l2_features.register_hook(lambda grad: self.activations_hook(grad, 'sa2'))
        # print(f"SA2 - coords: {l2_xyz.shape}, features: {l2_features.shape}")

        # SA3
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        self.features['sa3'] = l3_features
        if l3_features.requires_grad:
            h = l3_features.register_hook(lambda grad: self.activations_hook(grad, 'sa3'))
        # print(f"SA3 - coords: {l3_xyz.shape}, features: {l3_features.shape}")

        # Classification
        global_features = l3_features.view(B, -1)
        return self.classifier(global_features)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1) if points is not None else None

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


# Interpolate saliency to the original points
def interpolate_saliency(original_xyz, sampled_xyz, sampled_saliency, radius=0.1):
    assert original_xyz.device == sampled_xyz.device == sampled_saliency.device, \
        f"All tensors are on the same device: {original_xyz.device}, {sampled_xyz.device}, {sampled_saliency.device}"
    """
    original_xyz: [N, 3]
    sampled_xyz: [S, 3]
    sampled_saliency: [S]
    """
    device = original_xyz.device
    # Distance matrix [N, S]
    dist = torch.cdist(original_xyz, sampled_xyz, p=2.0)

    k = min(15, sampled_xyz.shape[0])
    knn_dist, knn_idx = torch.topk(dist, k=k, dim=1, largest=False)

    # Calculate weights
    weights = 1.0 / (knn_dist + 1e-6)  # [1, N, k]
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Collect and weight saliency
    gathered_saliency = sampled_saliency[knn_idx]  # [N, k]
    interpolated = (weights * gathered_saliency).sum(dim=1)  # [N]

    return interpolated


# Calculate multi-layer Grad-CAM
def compute_gradcam(model, input_pc, original_xyz_all, weights=muti_weights, smooth_kernel_size=5):
    model.eval()
    model.zero_grad()
    input_pc.requires_grad_(True)
    output = model(input_pc)
    pred_class = output.argmax(1).item()
    output[0, pred_class].backward(retain_graph=True)
    saliency_maps = []
    device = input_pc.device

    all_sampled_xyz = []  # Save sample points of each layer

    for i, name in enumerate(['sa1', 'sa2', 'sa3']):
        if name not in model.gradients:
            continue

        features = model.features[name].detach()
        grads = model.gradients[name].detach()
        xyz = model.features.get(f"{name}_xyz", None)

        # Save sampled coords
        if xyz is not None:
            all_sampled_xyz.append(xyz[0].permute(1, 0))  # [N, 3]
        # Smooth gradient
        if smooth_kernel_size > 1:
            smoothed_grads = []
            kernel = torch.ones(1, 1, smooth_kernel_size, device=grads.device) / smooth_kernel_size
            for c in range(grads.shape[1]):
                channel_grad = grads[:, c:c + 1, :]
                smoothed = F.conv1d(channel_grad, kernel, padding=smooth_kernel_size // 2)
                smoothed_grads.append(smoothed)
            grads = torch.cat(smoothed_grads, dim=1)
        # Saliency
        weights_per_channel = torch.mean(grads, dim=2, keepdim=True)
        saliency = F.relu(weights_per_channel * features)
        saliency = saliency.mean(dim=1).squeeze(0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency = torch.log(1 + torch.exp(2 * (saliency - 0.5)))
        if name == 'sa3':
            saliency = torch.ones_like(input_pc[0, :, 0]) * saliency.mean()
        elif xyz is not None:
            saliency = interpolate_saliency(
                input_pc[0, :, :3],
                xyz[0].permute(1, 0),
                saliency
            )
        saliency_maps.append(saliency * weights[i])
    if 'sa3' in model.gradients:
        # Return gradients of sa3
        sa3_grad = model.gradients['sa3']  # (1, C, 1)
        sa3_feat = model.features['sa3']  # (1, C, 1)


        sa3_saliency = (sa3_grad * sa3_feat).mean()  # Global importance

        # Access saliency on the sa2
        sa2_saliency = saliency_maps[1] if len(saliency_maps) >= 2 else None

        if sa2_saliency is not None:
            # Backward propagation to sa2
            sa3_mapped = sa3_saliency * torch.softmax(sa2_saliency, dim=0)
        else:
            sa3_mapped = torch.ones_like(input_pc[0, :, 0]) * sa3_saliency
        saliency_maps.append(sa3_mapped*weights[-1])
    saliency_maps = [sm.to(device) for sm in saliency_maps]
    # Concat multi-layer saliency
    final_saliency = torch.stack(saliency_maps).sum(0)
    final_saliency = (final_saliency - final_saliency.min()) / \
                     (final_saliency.max() - final_saliency.min() + 1e-8)
    # Interpolate to the original point cloud
    device = final_saliency.device
    original_xyz_all_tensor = torch.from_numpy(original_xyz_all).float().to(device)
    downsampled_xyz = input_pc[0, :, :3].to(device)  # Downsampled coords

    full_saliency = interpolate_saliency(
        original_xyz_all_tensor,  # Original xyz
        downsampled_xyz,  # Down sampled points
        final_saliency  # Down sampled saliency
    )

    return [full_saliency.cpu()], pred_class

def load_pointcloud(file_path, num_points=4096):
    try:
        points = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
    except ValueError:
        points = np.loadtxt(file_path, dtype=np.float32)

    # Extract coords and conduct normalization
    original_xyz_all = points[:, :3].copy()  # Keep initial points
    centroid = np.mean(original_xyz_all, axis=0)
    original_xyz_all -= centroid
    max_dist = np.max(np.sqrt(np.sum(original_xyz_all ** 2, axis=1)))
    original_xyz_all /= (max_dist + 1e-6)

    normalized_xyz = original_xyz_all.copy()
    xyz_tensor = torch.from_numpy(normalized_xyz).float().unsqueeze(0)  # [1, N, 3]
    fps_idx = farthest_point_sample(xyz_tensor, num_points)  # [1, num_points]
    sampled_xyz = index_points(xyz_tensor, fps_idx)
    if points.shape[1] > 3:
        color_tensor = torch.from_numpy(points[:, 3:]).float().unsqueeze(0)  # [1, N, C]
        sampled_color = index_points(color_tensor, fps_idx)  # [1, num_points, C]
        sampled_points = torch.cat([sampled_xyz, sampled_color], dim=-1)
    else:
        sampled_points = sampled_xyz
    return sampled_points, original_xyz_all, (centroid, max_dist)


# Visualize and save point cloud with saliency
def visualize_saliency(original_xyz_all, centroid, max_dist, saliency,
                      pred_class, input_file_path, output_dir):

    original_xyz_restored = original_xyz_all * max_dist + centroid
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    saliency_np = saliency.detach().cpu().numpy() if saliency.requires_grad else saliency

    # Normalize saliency
    min_val = np.percentile(saliency_np, 2)
    max_val = np.percentile(saliency_np, 98)
    saliency_np = (saliency_np - min_val) / (max_val - min_val + 1e-8)
    saliency_np = np.clip(saliency_np, 0, 1)
    # Heatmap_colors
    colors = plt.cm.jet(saliency_np.reshape(-1, 1))[:, 0, :3]

    # Draw point cloud
    scatter = ax.scatter(
        original_xyz_restored[:, 0],
        original_xyz_restored[:, 1],
        original_xyz_restored[:, 2],
        c=colors,
        s=3,
        alpha=0.9
    )
    plt.title(f"Class {pred_class} - Saliency Map")
    plt.colorbar(scatter, ax=ax, label='Attention')
    # Save image
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_img = os.path.join(output_dir, f"{file_name}_saliency.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved results to: {output_img}")

    output_pc = os.path.join(output_dir, f"{file_name}_saliency.txt")

    with open(input_file_path, 'r') as f:
        first_line = f.readline().strip()
    num_features = len(first_line.split(',')) - 3

    original_data = np.loadtxt(input_file_path, delimiter=',')
    saliency_col = saliency_np.reshape(-1, 1)

    if original_data.shape[1] > 3:
        output_data = np.hstack([original_data[:, :3],
                                 original_data[:, 3:],
                                 saliency_col])
    else:
        output_data = np.hstack([original_data[:, :3], saliency_col])

    # Save vile
    header = "x,y,z"
    if original_data.shape[1] > 3:
        header += "," + ",".join([f"f{i}" for i in range(original_data.shape[1] - 3)])
    header += ",saliency"

    np.savetxt(output_pc, output_data, delimiter=',', fmt='%.6f' , comments='')
    print(f"Saving original file to: {output_pc}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    # Load model
    model_path = r"D:\MASTERS\PGCAP__master\model_pth\best_model.pth"
    output_dir = r"E:\sensat_ply\PGCAP__master\predicted_result"
    test_folder = r"E:\sensat_ply\PGCAP__master\data\all_to_predict"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = PointNet2ClsWithHooks(num_classes=3).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    print(f"\nProcessing folder: {test_folder}")
    for file_name in sorted(os.listdir(test_folder)):
        if not file_name.endswith('.txt'):
            continue
        file_path = os.path.join(test_folder, file_name)
        print(f"\n========== Processing file: {file_name} ==========")
        try:
            print("Load point cloud...")
            tensor_pc, original_xyz_all, (centroid, max_dist) = load_pointcloud(file_path)
            print(f"Original points: {original_xyz_all.shape[0]}")
            print(f"Down sampled to: {tensor_pc.shape[1]}个点")

            tensor_pc = tensor_pc.to(device)
            print("\nCalculating Grad-CAM...")
            saliency_maps, pred_class = compute_gradcam(model, tensor_pc, original_xyz_all)

            print("\nGenerating visualization...")
            visualize_saliency(
                original_xyz_all, centroid, max_dist,
                saliency_maps[-1], pred_class, file_path, output_dir
            )
        except Exception as e:
            print(f"Processing {file_name} error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()




