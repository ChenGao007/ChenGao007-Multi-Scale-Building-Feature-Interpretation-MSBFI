import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from tqdm import tqdm
from datetime import datetime
import cv2
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

#SA1_weight for building details (wall texture; window frame; etc )
#SA2_weight for building components (window; doors; roof facet; roof ridge; etc)
#SA3_weight for general shape (building shape; outline; etc)

SA1_weight=0.4
SA2_weight=0.4
SA3_weight=0.2

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

        # SA1: input (xyz+rgb)
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=6,  # xyz+rgb
            mlp=[64, 64, 128],
            use_bn=False
        )

        # SA2: 128+3
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256]
        )

        # SA3: global feature
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024]
        )

        # model processing
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

        # Extract shape feature
        xyz, features = self.sa1(xyz, features)  # (B, 128, 512)
        xyz, features = self.sa2(xyz, features)  # (B, 256, 128)
        _, features = self.sa3(xyz, features)  # (B, 1024, 1)

        # Debug
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"\n[Debugging model internal features]")
            print(
                f"SA1 Output shape: {features.shape if isinstance(features, torch.Tensor) else [f.shape for f in features]}")
            print(f"SA3 Output maximum feature: {features.max().item():.4f}, minimum feature: {features.min().item():.4f}")
            print(f"SA3 Output non-zero feature proportion: {(features > 0).float().mean().item() * 100:.2f}%")

        # Global feature
        x = torch.max(features, 2)[0]  # (B, 1024)

        # Classification header
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class MultiLayerGradCAM:
    def __init__(self, model, layer_weights=None):
        self.model = model.eval()
        self.layer_weights = layer_weights or {"sa1": SA1_weight, "sa2": SA2_weight, "sa3": SA2_weight}  # 可调权重
        self.gradients = {}  # Save gradients of every layers
        self.activations = {}  # Save activations of every layers
        self.handles = []  # Hook handles
        self.debug_mode = True

        # Register hooks of SA1/SA2/SA3
        self._register_layer_hooks()

    def _register_layer_hooks(self):
        """Register hooks of SA1、SA2、SA3"""
        target_layers = {
            "sa1": "sa1.mlp_convs.2",  # Last convolution layer of SA1
            "sa2": "sa2.mlp_convs.2",  # Last convolution layer of SA2
            "sa3": "sa3.mlp_convs.2"  # Last convolution layer of SA3
        }

        def forward_hook_factory(layer_name):
            def forward_hook(module, input, output):
                self.activations[layer_name] = output.detach()
                if self.debug_mode:
                    print(f"\n[Forward activation] {layer_name}")
                    print(f"Shape: {output.shape}")
                    print(f"Value range: {output.min().item():.4f}~{output.max().item():.4f}")

            return forward_hook

        def backward_hook_factory(layer_name):
            def backward_hook(module, grad_input, grad_output):
                current_grad = grad_output[0].detach()
                self.gradients[layer_name] = current_grad * 10.0  # 梯度放大
                if self.debug_mode:
                    print(f"\n[Backward gradient] {layer_name}")
                    print(f"Shape: {current_grad.shape}")
                    print(f"Value range: {current_grad.min().item():.6f}~{current_grad.max().item():.6f}")

            return backward_hook

        # Register all hooks
        for name, layer_path in target_layers.items():
            layer = self._find_layer(layer_path)
            if layer is None:
                raise ValueError(f"Layer {layer_path} not found")

            self.handles.append(layer.register_forward_hook(forward_hook_factory(name)))
            self.handles.append(layer.register_full_backward_hook(backward_hook_factory(name)))

    def _find_layer(self, layer_name):
        """Find target layer"""
        modules = dict([*self.model.named_modules()])
        return modules.get(layer_name, None)

    def get_cam(self, point_cloud, target_class=None):
        """Calculate CAM of multiple layers"""
        B, C, N = point_cloud.shape

        # Forward propagation
        output = self.model(point_cloud)
        pred_class = output.argmax().item() if target_class is None else target_class

        # Backward propagation
        one_hot = torch.zeros_like(output)
        one_hot[0][pred_class] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        # Check gradient
        if not self.gradients:
            raise RuntimeError("No gradients captured! Check hook registration.")

        # Multi-layer CAM combination
        combined_cam = torch.zeros(N).to(point_cloud.device)
        for layer in ["sa1", "sa2", "sa3"]:
            if layer not in self.activations or layer not in self.gradients:
                continue

            # Calculate weight of the layer
            weights = torch.mean(self.gradients[layer], dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * self.activations[layer], dim=1).squeeze()

            # Normalized and weighted
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            combined_cam += cam * self.layer_weights[layer]

            if self.debug_mode:
                print(f"\n[{layer} CAM statistics]")
                print(f"Weight: {self.layer_weights[layer]:.2f}")
                print(f"Raw CAM range: {cam.min().item():.4f}~{cam.max().item():.4f}")

        # Last combination
        combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
        combined_cam = combined_cam.cpu().numpy()

        if self.debug_mode:
            print("\n[Final Combined CAM]")
            print(f"Shape: {combined_cam.shape}")
            print(f"Value range: {combined_cam.min():.4f}~{combined_cam.max():.4f}")
            print(f"Non-zero points: {(combined_cam > 0.01).sum() / len(combined_cam) * 100:.2f}%")

        return combined_cam, pred_class

    def __del__(self):
        """Remove """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def apply_saliency_smoothing(saliency, xyz, k=5):
    """Spatial distance-based saliency smoothing"""
    from sklearn.neighbors import NearestNeighbors

    # Find the k nearest neighbors of each point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    # Kernel function weights (the closer the distance the greater the weight)
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Neighborhood significance weighted average
    smoothed_saliency = np.zeros_like(saliency)
    for i in range(len(saliency)):
        smoothed_saliency[i] = np.sum(weights[i] * saliency[indices[i]])

    return smoothed_saliency


def visualize_saliency(xyz, saliency, save_path=None):
    """Visualizing saliency map"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=saliency, cmap='jet', s=5,
        vmin=0, vmax=1
    )
    plt.colorbar(sc, label='Saliency Value')
    plt.title('Point Cloud Saliency Map')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def percentile_clip(saliency_map, min_percentile=0, max_percentile=100):

    min_val = np.percentile(saliency_map, min_percentile)
    max_val = np.percentile(saliency_map, max_percentile)

    saliency_map = np.clip(saliency_map, min_val, max_val)
    saliency_map = (saliency_map - min_val) / (max_val - min_val + 1e-8)

    return saliency_map


def apply_gaussian_blur(saliency_map, sigma=1.0):
    """Applying Gaussian blur to increase smoothness"""

    # Blurring the saliency values as weights of the point cloud density
    return gaussian_filter(saliency_map, sigma=sigma)


def generate_demo_point_cloud(num_points):
    """Generating test point clouds"""
    xyz = np.random.randn(num_points, 3) * 5
    rgb = np.random.rand(num_points, 3) * 0.5 + 0.5
    return np.concatenate([xyz, rgb], axis=1)


def run_gradcam(input_path, output_dir, model_path):
    """Main Grad-CAM predicting process"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 50}")
    print(f"Processing: {input_path}")

    try:
        # 1. Loading data
        if isinstance(input_path, np.ndarray):
            data = input_path
        else:
            data = np.loadtxt(input_path, delimiter=',')

        xyz = data[:, :3]
        rgb = data[:, 3:] / 255.0 if data.shape[1] > 3 else np.zeros((len(xyz), 3))
        num_points = len(xyz)

        print(f"Point number of point cloud: {num_points}")

        # 2. Centralizing coordinates
        centroid = np.mean(xyz, axis=0)
        xyz_normalized = xyz - centroid

        # 3. Converting to PyTorch tensor
        point_cloud = torch.FloatTensor(
            np.vstack([xyz_normalized.T, rgb.T])  # (6, N)
        ).unsqueeze(0).to(device)

        print(f"Input tensor shape: {point_cloud.shape}")

        # 4. Loading model
        model = PointNet2Cls().to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully！")
        except Exception as e:
            raise RuntimeError(f"Model loading failure: {str(e)}")

        # 5. Calculating saliency map
        gcam = MultiLayerGradCAM(model)
        saliency, pred_class = gcam.get_cam(point_cloud)
        # Applying spatial smoothing
        saliency = apply_saliency_smoothing(saliency, xyz_normalized, k=10)
        print(f"Non-zero salient point proportion: {(saliency > 0.01).sum() / len(saliency) * 100:.2f}%")
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("Current time:", formatted_time)
        # 6. Saving result
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(input_path))[0] if isinstance(input_path, str) else "demo"

            # Visualization
            vis_path = os.path.join(output_dir, f"{filename}_saliency.jpg")
            visualize_saliency(xyz, saliency, save_path=vis_path)

            # Point cloud with saliency values
            output_data = np.hstack([data, saliency[:, None]])
            np.savetxt(
                os.path.join(output_dir, f"{filename}_saliency.txt"),
                output_data,
                delimiter=','
            )

            print(f"Saved to: {os.path.abspath(output_dir)}")

        return saliency, pred_class

    except Exception as e:
        print(f"Processing failure: {str(e)}")
        raise


def generate_debug_point_cloud(num_points, name):
    """generate test point cloud for debug"""
    xyz = np.random.randn(num_points, 3) * 5
    rgb = np.random.rand(num_points, 3) * 0.5 + 0.5

    print(f"\n⭐ Generated {name} point cloud:")
    print(f"XYZ shape: {xyz.shape}, Range: {xyz.min():.2f}~{xyz.max():.2f}")
    print(f"RGB shape: {rgb.shape}, Range: {rgb.min():.2f}~{rgb.max():.2f}")

    return np.concatenate([xyz, rgb], axis=1)

def main():
    # Generating test point cloud
    test_point_clouds = {
        "small": generate_debug_point_cloud(5000, "Small"),
        "medium": generate_debug_point_cloud(20000, "Medium"),
        "large": generate_debug_point_cloud(50000, "Large")
    }
    # Designation of paths
    MODEL_PATH = "./model_pth/best_model.pth"
    OUTPUT_DIR = "./predicted_result"

    # Testing three point clouds of different sizes
    for name, pc in test_point_clouds.items():
        print(f"\n{'=' * 50}")
        print(f"Testing point cloud: {name} ({len(pc)} points)")
        try:
            saliency, pred_class = run_gradcam(pc, OUTPUT_DIR, MODEL_PATH)
            print(f"Success！Predicted label: {pred_class}")
        except Exception as e:
            print(f"Testing failure: {str(e)}")

    # Actual batch processing
    input_folder = "./data/all_to_predict"
    if os.path.exists(input_folder):
        print(f"\nStart processing: {input_folder}")
        file_list = glob(os.path.join(input_folder, '*.txt'))
        for file_path in tqdm(file_list, desc="Processing"):
            try:
                run_gradcam(file_path, OUTPUT_DIR, MODEL_PATH)
            except Exception as e:
                print(f"\nProcessing failure: {file_path}\nerror occured in: {str(e)}")


if __name__ == "__main__":
    main()
