import torch
import numpy as np
import time
import os
import re
import shutil
import ast
import glob
import pandas as pd


class GenericUtils:
    def __init__(self):
        pass

    def set_seed(self, seed, print_seed=False):
        """
        Sets a random seed throughout the notebook.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if print_seed:
            print(f"Random seed set as {seed}")
            print(f"Random scalar: {np.random.rand()}")

    def set_device(self, device):
        return torch.device(device)

    def time_operation(self, start, message):
        end = time.perf_counter()
        elapsed = end - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"{message}: {minutes} min {seconds:.2f} sec")
        # return minutes, seconds

    def _get_highest_sorted_file(self, directory, prefix, extension):
        # Gather all CSV files starting with prefix and located in directory
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.startswith(prefix)
            and f.endswith(extension)
        ]

        if not files:
            return None

        # Sort files in descending order and return the first
        highest_file = sorted(files, reverse=True)[0]
        filename = directory + "/" + highest_file
        return filename

    def _extract_part_timestamp(self, filename):
        pattern = r"^(part_\d+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
        match = re.match(pattern, filename)
        return match.group(1) if match else None

    def rename_and_move_image(self, local_file, data_directory, prefix, extension, image_directory):
        # Get filename
        csv_full_file = self._get_highest_sorted_file(data_directory, prefix, extension)
        csv_file_name = csv_full_file.split("/")[-1].split(".")[0]

        file_prefix = self._extract_part_timestamp(csv_file_name)
        new_filename = f"{image_directory}/{file_prefix}_anim.png"

        # Rename the file locally (if needed)
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"File {local_file} not found!")

        # Copy file to new location
        shutil.copy(local_file, new_filename)
        return new_filename

    def add_score_to_df(self, score, grade, directory, prefix, extension):
        """
        Add score and grade to CSV.
        """
        filename = self._get_highest_sorted_file(directory, prefix, extension)
        df = pd.read_csv(filename, index_col=None)

        try:
            df.insert(1, "score", [round(score, 6)])
            df.insert(2, "grade", [round(grade, 6)])
            df.to_csv(filename, index=False)
        except:
            print("'score' and 'grade' already exist as columns")

        return df

    def _combine_csv_files(self, directory, prefix, extension=".csv"):
        search_pattern = os.path.join(directory, f"{prefix}*{extension}")
        csv_files = glob.glob(search_pattern)

        if not csv_files:
            print("No matching CSV files found.")
            return None

        df_combined = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f"Combined {len(csv_files)} files.")
        return df_combined

    def view_metrics(
        self, data_directory, image_directory, plots_directory, prefix, scoring_boundary
    ):
        df = self._combine_csv_files(data_directory, prefix)
        df = df.sort_values(by=["grade"], ascending=False).reset_index(drop=True)

        if df is not None:
            cols_to_view = [
                col
                for col in df.columns
                if col not in ["model_features", "model_architecture", "optimizer", "scheduler"]
            ]
            print(df[cols_to_view].sort_values(by=["grade"], ascending=False).head())

        print("\n============================== Filename ==============================")
        print(df["file_name"][0])
        print("\n============================== Model Architecture ==============================")
        print(df["model_architecture"][0])

        try:
            feature_list = ast.literal_eval(df["model_features"][0])
            print(f"\n============================== Model Features ==============================")
            for feature in feature_list:
                print(feature)
        except:
            print()

        print("\n============================== Optimizer ==============================")
        print(df["optimizer"][0])

        print("\n============================== Scheduler ==============================")
        print(df["scheduler"][0])

        print("\n============================== Scoring ==============================")
        print(f"Score: {df['score'][0]}")
        print(f"Grade: {df['score'][0]} / {scoring_boundary:.2f} * 4 = {df['grade'][0]:.2f}")
