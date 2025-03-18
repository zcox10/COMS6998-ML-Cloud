import os
import torch
from utils.utils import Utils
from datetime import datetime


class ModelTwo(torch.nn.Module):
    def __init__(self, device_str):
        super(ModelTwo, self).__init__()
        self.utils = Utils()
        self.device_str = device_str

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=128 * 8 * 8, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=256, out_features=4),
        )

    def forward(self, x):
        return self.net(x)

    def model_parameters(self):
        return {
            "batch_size": 64,
            "epochs": 50,
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001),
            "scheduler": None,
            "patience": 10,
            "clip_value": 1.0,
            "device": self.utils.set_device(self.device_str),
            "early_stop": True,
        }

    def data_processing(self, batch_size):
        # Load data
        self.utils.download_data()
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of model_training.py
        file_path = os.path.join(current_dir, "training_data/map1.pkl")
        data = self.utils.load_data(file_path)
        data.pop("poses")
        data.pop("agent")
        data["obs"] = data.pop("rgb")

        dataset_mean, dataset_std = self.utils.compute_image_dataset_statistics(data["obs"])

        # Create data loaders
        X = self.utils.numpy_images_to_pytorch_tensors(
            data["obs"], torch.float32, dataset_mean, dataset_std
        )
        y = self.utils.numpy_to_pytorch_tensors(data["actions"], torch.long)

        dataset = self.utils.create_tensor_dataset(X, y)
        train_loader, val_loader, test_loader = self.utils.create_data_loaders(
            dataset, batch_size=batch_size
        )
        return train_loader, val_loader, test_loader

    def train_model(self):
        model_params = self.model_parameters()
        train_loader, val_loader, _ = self.data_processing(model_params["batch_size"])
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
            input_shape=(model_params["batch_size"], 3, 64, 64),
            profile_model=True,
            roofline_model_save_file=f"results/plots/model-2-{gpu_name}-{current_timestamp}.png",
            training_metrics_save_file=f"results/data/model-2-{gpu_name}-{current_timestamp}.csv",
        )
