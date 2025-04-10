import os
import subprocess
import pickle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import torchvision.utils as vutils


class DatasetUtils:
    def __init__(self):
        pass

    def _create_training_transform(self, mean_vals, std_vals):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_vals, std=std_vals),
            ]
        )

    def _create_testing_transform(self, mean_vals, std_vals):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)]
        )

    def compute_image_dataset_statistics(self, data):
        """
        Compute mean and std of a dataset.
        """
        mean_vals = data.mean(axis=(0, 1, 2)).tolist()
        std_vals = data.std(axis=(0, 1, 2)).tolist()
        return mean_vals, std_vals

    def load_data(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def download_data(self):
        """
        Checks if map1.pkl and all_maps.pkl exists in ./models/training_data.
        If not, downloads project_data.zip, unzips it, and moves the pkl files.
        """
        map1_path = "./models/training_data/map1.pkl"
        all_maps_path = "./models/training_data/all_maps.pkl"

        # Check if either file is missing
        if not (os.path.exists(map1_path) and os.path.exists(all_maps_path)):
            print("Files not found. Downloading and extracting data...")

            # Download
            subprocess.run(
                [
                    "wget",
                    "https://www.dropbox.com/scl/fi/gy1d0ifkwuusmdjv796dl/project2.zip?rlkey=h6wresrsqxiryhlvrssjla5hn&st=cfvqccqm&dl=0",
                ]
            )

            # Rename
            subprocess.run(
                [
                    "mv",
                    "project2.zip?rlkey=h6wresrsqxiryhlvrssjla5hn&st=cfvqccqm&dl=0",
                    "project_data.zip",
                ]
            )

            # Unzip (overwrite if needed)
            subprocess.run(["unzip", "-o", "project_data.zip", "-d", "./project_data"])

            # Ensure target directory exists
            os.makedirs(os.path.dirname(map1_path), exist_ok=True)

            # Move files to final destination
            subprocess.run(["mv", "./project_data/data/map1.pkl", map1_path])
            subprocess.run(["mv", "./project_data/data/all_maps.pkl", all_maps_path])

            # Clean up
            subprocess.run(["rm", "-rf", "./project_data"])
            subprocess.run(["rm", "-rf", "./project_data.zip"])

            print("Download and extraction complete.")
        else:
            print("map1.pkl and all_maps.pkl already exist. Skipping download.")

    def print_class_counts(self, y):
        """
        Print class counts and pct of total labels.
        """
        class_counts = Counter(y.tolist())
        total_samples = len(y)

        class_counts_list = [[cls, count] for cls, count in class_counts.items()]
        class_counts_list.sort(key=lambda x: -x[1])

        for cls in class_counts_list:
            action = cls[0]
            count = cls[1]

            pct = round((count / total_samples) * 100, 2)
            print(f"Action {action}: {count} samples ({pct}%)")

    def numpy_images_to_pytorch_tensors(self, data, datatype, mean_vals, std_vals):
        """
        Convert a batch of NumPy images to PyTorch tensors and apply transformations.
        """
        train_transform = self._create_training_transform(mean_vals, std_vals)
        test_transform = self._create_testing_transform(mean_vals, std_vals)

        if isinstance(data, np.ndarray):
            if len(data.shape) == 4:  # Batch of images (B, H, W, C)
                data = torch.stack([train_transform(img) for img in data])
            elif len(data.shape) == 3:  # Single image (H, W, C)
                data = train_transform(data)

        return data.to(dtype=datatype)

    def numpy_to_pytorch_tensors(self, data, datatype):
        """
        Convert numpy arrays to PyTorch tensors.
        """
        return torch.tensor(data, dtype=datatype)

    def create_tensor_dataset(self, X, y):
        return TensorDataset(X, y)

    def create_data_loaders(self, dataset, batch_size):
        # Define split sizes
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def _standardize_features(self, features, mean, std):
        """
        Standardize features by subtracting the mean and dividing by the standard deviation.
        """
        return (features - mean) / std

    def _compute_dataset_statistics(self, features):
        """
        Compute statistics for the dataset, mean and std.
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1
        return mean, std

    def add_features_numpy(self, obs, feature_cols, dataset_mean, dataset_std):

        # Convert to 2D numpy array if a single observation
        is_single_obs = len(obs.shape) == 1
        obs = obs.reshape(1, -1) if is_single_obs else obs
        agent_x, agent_y, goal_x, goal_y = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]

        # Compute Manhattan distance
        manhattan_dist = np.abs(goal_x - agent_x) + np.abs(goal_y - agent_y)

        # Create hypothetical positions based on a given action
        next_0_x = agent_x.copy()
        next_0_y = agent_y.copy() + 0.1

        next_1_x = agent_x.copy() - 0.1
        next_1_y = agent_y.copy()

        next_2_x = agent_x.copy() + 0.1
        next_2_y = agent_y.copy()

        next_3_x = agent_x.copy()
        next_3_y = agent_y.copy() - 0.1

        # Compute manhattan distance
        norm1_dist_0 = np.abs(goal_x - next_0_x) + np.abs(goal_y - next_0_y)
        norm1_dist_1 = np.abs(goal_x - next_1_x) + np.abs(goal_y - next_1_y)
        norm1_dist_2 = np.abs(goal_x - next_2_x) + np.abs(goal_y - next_2_y)
        norm1_dist_3 = np.abs(goal_x - next_3_x) + np.abs(goal_y - next_3_y)

        # Compute relative position
        dx = goal_x - agent_x
        dy = goal_y - agent_y

        feature_dict = {
            "agent_x": agent_x,
            "agent_y": agent_y,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "manhattan_dist": manhattan_dist,
            "norm1_dist_0": norm1_dist_0,
            "norm1_dist_1": norm1_dist_1,
            "norm1_dist_2": norm1_dist_2,
            "norm1_dist_3": norm1_dist_3,
            "next_0_x": next_0_x,
            "next_0_y": next_0_y,
            "next_1_x": next_1_x,
            "next_1_y": next_1_y,
            "next_2_x": next_2_x,
            "next_2_y": next_2_y,
            "next_3_x": next_3_x,
            "next_3_y": next_3_y,
        }
        final_dict = {k: v for k, v in feature_dict.items() if k in feature_cols}

        # Stack selected features
        feature_values = np.column_stack(list(final_dict.values()))

        # If not a single observation, then we are training. So, calculate mean/std and store
        if not is_single_obs:
            dataset_mean, dataset_std = self._compute_dataset_statistics(feature_values)

        standardized_obs = self._standardize_features(feature_values, dataset_mean, dataset_std)
        return standardized_obs, dataset_mean, dataset_std

    def show_sample_images(self, dataloader, title, mean_vals, std_vals, num_images):
        """
        Displays a batch of images from the given dataloader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader to sample from.
            title (str): Title of the plot.
            mean_vals (list): Mean values used for normalization.
            std_vals (list): Std values used for normalization.
            num_images (int): Number of images to display.
        """
        # Get one batch of data
        images, labels = next(iter(dataloader))

        # Denormalize images for better visualization
        inv_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-m / s for m, s in zip(mean_vals, std_vals)],
                    std=[1 / s for s in std_vals],
                )
            ]
        )
        images = torch.stack([inv_transform(img) for img in images])  # Apply inverse normalization

        # Make grid of images
        grid = vutils.make_grid(images[:num_images], nrow=4, normalize=True)

        # Convert from Tensor to NumPy and plot
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))  # Convert CHW to HWC for Matplotlib
        plt.axis("off")
        plt.title(title)
        plt.show()
