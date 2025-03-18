import os
import torch
from utils.utils import Utils
from datetime import datetime


class ModelOne(torch.nn.Module):
    def __init__(self, device_str):
        super(ModelOne, self).__init__()
        self.utils = Utils()
        self.device_str = device_str

        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)

    def model_parameters(self):

        # Feature columns to keep
        feature_cols = [
            "agent_x",
            "agent_y",
            "goal_x",
            "goal_y",
            "manhattan_dist",
            "norm1_dist_0",
            "norm1_dist_1",
            "norm1_dist_2",
            "norm1_dist_3",
        ]

        return {
            "batch_size": 100,
            "epochs": 128,
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.AdamW(self.parameters(), lr=0.004),
            "scheduler": None,
            "patience": 5,
            "clip_value": 1.0,
            "device": self.utils.set_device(self.device_str),
            "early_stop": False,
            "feature_cols": feature_cols,
        }

    def data_processing(self, feature_cols, batch_size):
        # Load data
        self.utils.download_data()
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of model_training.py
        file_path = os.path.join(current_dir, "training_data/map1.pkl")
        data = self.utils.load_data(file_path)
        data.pop("rgb")
        data.pop("agent")
        data["obs"] = data.pop("poses")

        # Generate features on data, generate X, y
        obs, _, _ = self.utils.add_features_numpy(data["obs"], feature_cols, None, None)
        X = self.utils.numpy_to_pytorch_tensors(obs, torch.float32)
        y = self.utils.numpy_to_pytorch_tensors(data["actions"], torch.long)

        # Convert to train, val, and test data loaders
        dataset = self.utils.create_tensor_dataset(X, y)
        train_loader, val_loader, test_loader = self.utils.create_data_loaders(
            dataset, batch_size=batch_size
        )
        return train_loader, val_loader, test_loader

    def train_model(self):
        model_params = self.model_parameters()
        train_loader, val_loader, _ = self.data_processing(
            model_params["feature_cols"], model_params["batch_size"]
        )
        model_params["train_loader"] = train_loader
        model_params["val_loader"] = val_loader
        print(f"Device: {model_params['device']}")

        gpu_name = (torch.cuda.get_device_properties("cuda").name).replace(" ", "-")
        current_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        training_metrics = self.utils.train_loop(
            epochs=model_params["epochs"],
            model=self,
            criterion=model_params["criterion"],
            optimizer=model_params["optimizer"],
            scheduler=model_params["scheduler"],
            train_dataloader=model_params["train_loader"],
            val_dataloader=model_params["val_loader"],
            patience=model_params["patience"],
            clip_value=model_params["clip_value"],
            device=model_params["device"],
            monitor_gradients=False,
            early_stop=model_params["early_stop"],
            input_shape=(model_params["batch_size"], 9),
            profile_model=True,
            roofline_model_save_file=f"results/plots/model-1-{gpu_name}-{current_timestamp}.png",
            training_metrics_save_file=f"results/data/model-1-{gpu_name}-{current_timestamp}.csv",
        )
