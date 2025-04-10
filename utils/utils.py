from utils.generic_utils import GenericUtils
from utils.dataset_utils import DatasetUtils
from utils.model_utils import ModelUtils


class Utils:
    def __init__(self):
        self.generic_utils = GenericUtils()
        self.generic_utils.set_seed(seed=1, print_seed=False)  # set seed
        self.dataset_utils = DatasetUtils()
        self.model_utils = ModelUtils()

    # GenericUtils
    def set_seed(self):
        self.generic_utils.set_seed()

    def set_device(self, device):
        return self.generic_utils.set_device(device)

    def time_operation(self, start, message="Elapsed time"):
        self.generic_utils.time_operation(start, message)

    def rename_and_move_image(self, local_file, data_directory, prefix, extension, image_directory):
        return self.generic_utils.rename_and_move_image(
            local_file, data_directory, prefix, extension, image_directory
        )

    def add_score_to_df(self, score, grade, directory, prefix, extension):
        return self.generic_utils.add_score_to_df(score, grade, directory, prefix, extension)

    def view_metrics(
        self, data_directory, image_directory, plots_directory, prefix, scoring_boundary
    ):
        return self.generic_utils.view_metrics(
            data_directory, image_directory, plots_directory, prefix, scoring_boundary
        )

    # DatasetUtils
    def load_data(self, path):
        return self.dataset_utils.load_data(path)

    def download_data(self):
        self.dataset_utils.download_data()

    def print_class_counts(self, y):
        self.dataset_utils.print_class_counts(y)

    def compute_image_dataset_statistics(self, data):
        return self.dataset_utils.compute_image_dataset_statistics(data)

    def numpy_images_to_pytorch_tensors(self, data, datatype, mean_vals, std_vals):
        return self.dataset_utils.numpy_images_to_pytorch_tensors(
            data, datatype, mean_vals, std_vals
        )

    def numpy_to_pytorch_tensors(self, data, datatype):
        return self.dataset_utils.numpy_to_pytorch_tensors(data, datatype)

    def create_tensor_dataset(self, X, y):
        return self.dataset_utils.create_tensor_dataset(X, y)

    def create_image_data_loaders(self, dataset, batch_size, mean_vals, std_vals):
        return self.dataset_utils.create_image_data_loaders(
            dataset, batch_size, mean_vals, std_vals
        )

    def create_data_loaders(self, dataset, batch_size):
        return self.dataset_utils.create_data_loaders(dataset, batch_size)

    def add_features_numpy(self, data, feature_cols, dataset_mean, dataset_std):
        return self.dataset_utils.add_features_numpy(data, feature_cols, dataset_mean, dataset_std)

    def show_sample_images(self, dataloader, title, mean_vals, std_vals, num_images):
        self.dataset_utils.show_sample_images(dataloader, title, mean_vals, std_vals, num_images)

    # ModelUtils
    def train_loop(self, **kwargs):
        return self.model_utils.train_loop(**kwargs)

    def evaluate_model(self, **kwargs):
        return self.model_utils.evaluate_model(**kwargs)
