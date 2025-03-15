import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.profiler
import time
import matplotlib.pyplot as plt
import numpy as np
import psutil
import subprocess


class ModelUtils:
    def __init__(self, save_dir="../results"):
        self.save_dir = save_dir

    def train_loop(
        self,
        epochs,
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        patience,
        clip_value,
        device,
        monitor_gradients,
        early_stop,
        input_shape,
        profile_model,
        roofline_model_save_file=None,
        training_metrics_save_file=None,
    ):
        """
        Training loop for an NN. Returns metrics for plotting and classification reports.

        Returns:
        - training_losses, validation_losses
        - train_labels, train_predictions
        - val_labels, val_predictions
        """

        start_training_time = time.perf_counter()
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0  # early stopping

        # Store all training predictions and labels
        train_predictions, train_labels = [], []
        val_predictions, val_labels = [], []

        # Move model to specified device
        model.to(device)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            # Training loop
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device).long()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # gradient clipping
                optimizer.step()
                total_loss += loss.item()

                # Compute accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                # Store predictions and labels
                train_predictions.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Compute training loss and accuracy
            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = correct_predictions / total_samples
            train_losses.append(avg_train_loss)

            # Run validation and store results
            avg_val_loss, val_accuracy, epoch_val_predictions, epoch_val_labels = (
                self._validation_loop(model, criterion, val_dataloader, device)
            )
            val_losses.append(avg_val_loss)
            val_predictions.extend(epoch_val_predictions)
            val_labels.extend(epoch_val_labels)

            # Early stopping logic
            if early_stop:
                patience_counter, best_val_loss = self._determine_early_stopping(
                    avg_val_loss, best_val_loss, patience_counter
                )
                # print(f"Patience counter: {patience_counter}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered! Epoch {epoch + 1}")
                    break

            # Print epoch summary
            if (epoch + 1) % 5 == 0:
                if monitor_gradients:
                    # Monitor for vanishing or exploding gradients
                    total_gradients = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_gradients.append(grad_norm)
                            if grad_norm > 1e6:
                                print(
                                    f"\nWarning: Exploding Gradient Detected in {name}: {grad_norm:.2e}\n"
                                )
                            elif grad_norm < 1e-8:
                                print(
                                    f"\nWarning: Vanishing Gradient Detected in {name}: {grad_norm}\n"
                                )
                    avg_grad = sum(total_gradients) / len(total_gradients) if total_gradients else 0

                progress_message = (
                    f"Epoch {epoch+1:>4}/{epochs:<4} | "
                    + f"Train Loss: {avg_train_loss:<7.4f} Train Acc: {train_accuracy:<7.4f} | "
                    + f"Val Loss: {avg_val_loss:<7.4f} Val Acc: {val_accuracy:<7.4f}"
                )
                progress_message += (
                    f" | Avg Gradient: {avg_grad:<10.8f}" if monitor_gradients else ""
                )
                print(progress_message)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif scheduler is not None:
                scheduler.step()

        # end training time
        end_training_time = time.perf_counter()
        samples_per_sec = (
            epochs
            * (len(train_dataloader) + len(val_dataloader))
            / (end_training_time - start_training_time)
        )

        if profile_model:
            total_flops, flops_per_sec = self._compute_flops(model, input_shape, device)
            total_bytes, mem_bytes_per_sec = self._compute_memory_bandwidth(
                model, train_dataloader, device
            )
            if total_bytes < 1e-8:
                print("WARNING: total_bytes is < 1e-8")
                total_bytes = 1e-8
            operational_intensity = total_flops / total_bytes

            peak_flops, peak_mem_bw = self._plot_roofline_model(
                device, total_flops, operational_intensity, roofline_model_save_file
            )

            self._save_training_metrics_df(
                data={
                    "model": [model.__class__.__name__],
                    "trainable_parameters": [
                        sum(p.numel() for p in model.parameters() if p.requires_grad)
                    ],
                    "device": [device],
                    "gpu": [torch.cuda.get_device_name(0) if torch.cuda.is_available() else None],
                    "total_gflops": [round(total_flops / 1e9, 6)],
                    "total_gb": [round(total_bytes / 1e9, 6)],
                    "gflops_per_sec": [round(flops_per_sec / 1e9, 6)],
                    "mem_bandwidth_gb_s": [round(mem_bytes_per_sec / 1e9, 6)],
                    "operational_intensity": [round(operational_intensity, 6)],
                    "peak_flops_gb": [round(peak_flops / 1e9, 4)],
                    "peak_mem_bw_gb_s": [round(peak_mem_bw / 1e9, 4)],
                    "samples_per_sec": [round(samples_per_sec, 2)],
                },
                training_metrics_save_file=training_metrics_save_file,
            )

        return {
            "train_labels": train_labels,
            "train_predictions": train_predictions,
            "train_losses": train_losses,
            "val_labels": val_labels,
            "val_predictions": val_predictions,
            "val_losses": val_losses,
            "train_accuracy": train_accuracy,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": avg_val_loss,
        }

    def _validation_loop(self, model, criterion, val_dataloader, device):
        """
        Validation loop and storing predictions.
        """

        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        val_predictions, val_labels = [], []

        # Validation loop
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0)

                # Store predictions and labels
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_samples

        return avg_val_loss, val_accuracy, val_predictions, val_labels

    def _determine_early_stopping(self, avg_val_loss, best_val_loss, patience_counter):
        """
        Checks if early stopping condition is met.
        """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        return patience_counter, best_val_loss

    def _calculate_test_performance(self, model, dataloader, criterion, device):
        """
        Evaluate a model on a test dataset to determine its performance.
        """
        model.to(device)

        model.eval()
        loss = 0
        correct, total = 0, 0
        test_predictions, test_labels = [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss += criterion(outputs, labels).item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                test_predictions.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        avg_loss = loss / len(dataloader)
        accuracy = correct / total

        print(f"\nTest Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}\n")
        return avg_loss, accuracy, test_predictions, test_labels

    def _plot_training_summary(
        self,
        train_losses,
        val_losses,
        train_labels,
        train_predictions,
        val_labels,
        val_predictions,
        test_labels,
        test_predictions,
        save_filename_prefix,
    ):
        """
        Creates a figure with:
        - Row 1: Loss curve
        - Row 2: Confusion matrices (Train, Validation, Test)
        - Row 3: Classification reports (Train, Validation, Test)
        """

        # 2 rows, 3 columns
        fig, axes = plt.subplots(3, 3, figsize=(18, 18), gridspec_kw={"height_ratios": [1, 1, 1]})
        fig.delaxes(axes[0, 0])
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[0, 2])

        # Merge first row into single wide plot for loss curves
        loss_ax = fig.add_subplot(3, 1, 1)

        # Plot loss curves
        epochs = np.arange(1, len(train_losses) + 1)
        loss_ax.plot(
            epochs, train_losses, color="b", marker="o", linestyle="-", label="Training Loss"
        )
        loss_ax.plot(
            epochs, val_losses, color="r", marker="o", linestyle="-", label="Validation Loss"
        )

        loss_ax.set_xlabel("Epochs")
        loss_ax.set_ylabel("Loss")
        loss_ax.set_title("Training and Validation Loss")
        loss_ax.legend()
        loss_ax.grid()

        if len(epochs) <= 50:
            xticks = np.arange(1, len(train_losses) + 1)
        else:
            xticks = np.arange(0, len(train_losses), 5)

        loss_ax.set_xticks(xticks)
        loss_ax.set_xticklabels(xticks, rotation=0)
        # loss_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Helper function to plot confusion matrix
        def _plot_conf_matrix(ax, labels, predictions, title):
            classes = np.sort(np.unique(np.concatenate((labels, predictions))))
            cm = confusion_matrix(labels, predictions, labels=classes)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)

        # Plot confusion matrices in second row
        _plot_conf_matrix(axes[1, 0], train_labels, train_predictions, "Train Confusion Matrix")
        _plot_conf_matrix(axes[1, 1], val_labels, val_predictions, "Validation Confusion Matrix")
        _plot_conf_matrix(axes[1, 2], test_labels, test_predictions, "Test Confusion Matrix")

        # Generate classification reports as text
        train_report = classification_report(train_labels, train_predictions)
        val_report = classification_report(val_labels, val_predictions)
        test_report = classification_report(test_labels, test_predictions)

        # Insert classification reports into the third row
        axes[2, 0].axis("off")  # Disable axes for text display
        axes[2, 0].text(
            0, 0.5, train_report, fontsize=10, va="center", ha="left", family="monospace"
        )
        axes[2, 0].set_title("Train Classification Report")

        axes[2, 1].axis("off")
        axes[2, 1].text(0, 0.5, val_report, fontsize=10, va="center", ha="left", family="monospace")
        axes[2, 1].set_title("Validation Classification Report")

        axes[2, 2].axis("off")
        axes[2, 2].text(
            0, 0.5, test_report, fontsize=10, va="center", ha="left", family="monospace"
        )
        axes[2, 2].set_title("Test Classification Report")

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save plot
        file_name = self.save_dir + f"plots/{save_filename_prefix}_training_summary.png"
        plt.savefig(file_name)
        plt.show()

    def evaluate_model(
        self,
        model,
        dataloader,
        criterion,
        device,
        training_metrics,
        model_parameters,
        save_filename_prefix,
    ):
        avg_test_loss, test_accuracy, test_predictions, test_labels = (
            self._calculate_test_performance(model, dataloader, criterion, device)
        )

        self._plot_training_summary(
            train_losses=training_metrics["train_losses"],
            val_losses=training_metrics["val_losses"],
            train_labels=training_metrics["train_labels"],
            train_predictions=training_metrics["train_predictions"],
            val_labels=training_metrics["val_labels"],
            val_predictions=training_metrics["val_predictions"],
            test_labels=test_labels,
            test_predictions=test_predictions,
            save_filename_prefix=save_filename_prefix,
        )

        # Save model parameters and performance to CSV
        df = pd.DataFrame([model_parameters])

        df.insert(0, "file_name", [save_filename_prefix])
        df.insert(1, "train_loss", [training_metrics["train_loss"]])
        df.insert(2, "val_loss", [training_metrics["val_loss"]])
        df.insert(3, "test_loss", [avg_test_loss])
        df.insert(4, "train_accuracy", [training_metrics["train_accuracy"]])
        df.insert(5, "val_accuracy", [training_metrics["val_accuracy"]])
        df.insert(6, "test_accuracy", [test_accuracy])

        df.to_csv(self.save_dir + f"data/{save_filename_prefix}_model_performance.csv", index=False)
        return df

    def _save_training_metrics_df(self, data, training_metrics_save_file):
        df = pd.DataFrame(data)
        df.to_csv(training_metrics_save_file, index=False)

    def _plot_roofline_model(
        self, device, total_flops, operational_intensity, roofline_model_save_file
    ):
        peak_flops = self._get_peak_flops(device)
        peak_mem_bw = self._get_peak_mem_bandwidth(device)

        # Define X-axis range in log scale
        OI_values = np.logspace(-2, 4, 100)

        # Compute roofline curve
        compute_bound = np.full_like(OI_values, peak_flops)  # Compute-bound horizontal line
        memory_bound = OI_values * peak_mem_bw  # Memory-bound sloped line

        plt.figure(figsize=(8, 6))
        plt.loglog(OI_values, np.minimum(compute_bound, memory_bound), "k-", label="Roofline")
        plt.scatter(operational_intensity, total_flops, color="red", label="Your Model", s=100)

        plt.xlabel("Operational Intensity (FLOPs/Byte)")
        plt.ylabel("Performance (FLOPs/s)")
        plt.title("Roofline Model for Your Neural Network")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(roofline_model_save_file)

        return peak_flops, peak_mem_bw

    def _compute_memory_bandwidth(self, model, data_loader, device):
        model.to(device)
        total_bytes = 0
        start_time = time.perf_counter()
        process = psutil.Process()

        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            # Measure before memory allocation
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                mem_start = torch.cuda.memory_allocated(device)
            else:
                mem_start = process.memory_info().rss

            # Forward Pass
            _ = model(inputs)

            # Measure after memory allocation
            if device == "cuda":
                torch.cuda.synchronize()
                mem_end = torch.cuda.max_memory_allocated(device)
            else:
                mem_end = process.memory_info().rss

            total_bytes += mem_end - mem_start

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Bandwidth
        return total_bytes, total_bytes / elapsed_time

    def _compute_flops(self, model, input_shape, device):
        model = model.to(device)
        dummy_input = torch.randn(*input_shape).to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + ([torch.profiler.ProfilerActivity.CUDA] if device == "cuda" else []),
            record_shapes=True,
            with_flops=True,
        ) as prof:
            model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
        flops_per_sec = total_flops / (end - start)
        return total_flops, flops_per_sec

    def _detect_avx(self):
        """Check for AVX/AVX2/AVX-512 support"""
        try:
            output = subprocess.check_output("lscpu", shell=True).decode()
            if "avx512f" in output:
                return 16  # AVX-512: 16 FLOPs per cycle
            elif "avx2" in output:
                return 8  # AVX2: 8 FLOPs per cycle
            elif "avx" in output:
                return 4  # AVX: 4 FLOPs per cycle
            else:
                return 2  # No AVX support
        except Exception as e:
            print(f"Error detecting AVX support: {e}")
            return 2

    def _get_peak_flops(self, device):
        if device == "cuda":
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            # Compute FLOP/s estimate
            cores_per_sm = {
                5.0: 128,
                5.2: 128,
                5.3: 128,  # Maxwell
                6.0: 64,
                6.1: 128,
                6.2: 128,  # Pascal
                7.0: 64,
                7.2: 64,
                7.5: 64,  # Volta & Turing
                8.0: 64,
                8.6: 128,
                8.9: 128,  # Ampere
                9.0: 128,  # Hopper
            }

            sm_count = props.multi_processor_count
            sm_architecture = props.major + props.minor / 10.0
            cores_per_sm_value = cores_per_sm.get(sm_architecture)

            clock_speed = props.clock_rate * 1e3
            peak_flops = sm_count * cores_per_sm_value * 2 * clock_speed

            return peak_flops

        else:
            avx_factor = self._detect_avx()
            cpu_freq = psutil.cpu_freq().max * 1e6
            num_cores = psutil.cpu_count(logical=True)

            peak_flops = num_cores * cpu_freq * avx_factor
            return peak_flops

    def _get_peak_mem_bandwidth(self, device):
        """Calculate peak memory bandwidth in bytes/s for CUDA or CPU device."""
        if device == "cuda":
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            return props.memory_clock_rate * 1e3 * props.memory_bus_width / 8
        else:
            # Run peak memory bandwidth benchmark for CPU
            size = 2**28
            a = np.random.randn(size).astype(np.float32)
            start_time = time.time()
            b = np.copy(a)
            end_time = time.time()
            return a.nbytes / (end_time - start_time)
