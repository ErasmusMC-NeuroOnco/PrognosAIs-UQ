import os
from timeit import default_timer as timer

import mlflow
import numpy as np
import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from prognosais.model.development.losses import dice_loss, masked_cross_entropy_loss
from prognosais.model.development.utils import EarlyStopping
from torchinfo import summary
from tqdm import tqdm


class Trainer:
    """
    Train CSNet and track losses, checkpoints, and MLflow artifacts.

    The trainer handles the segmentation and classification objectives used by
    the project. It can also run in classification-only mode, in which the
    segmentation decoder is bypassed and only the classification head is used.

    Attributes:
        model: Model being trained.
        modalities: Modalities used for training.
        num_epochs: Number of epochs to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        weights_idh: Class weights for the IDH task.
        weights_loss_seg: Relative weight for the segmentation loss.
        weights_loss_idh: Relative weight for the IDH loss.
        weights_1p19q: Class weights for the 1p19q task.
        weights_loss_1p19q: Relative weight for the 1p19q loss.
        weights_grade: Class weights for the grade task.
        weights_loss_grade: Relative weight for the grade loss.
        optimizer: Optimizer used for parameter updates.
        scheduler: Optional learning-rate scheduler.
        device: Device used for training and validation.
        config: Configuration wrapper loaded from the YAML file.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modalities: dict[str, str],
        num_epochs: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        weights_loss_seg: float,
        classification_weights: dict[str, torch.Tensor],
        classification_loss_weights: dict[str, float],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        config_dir: configIO.Config,
    ):
        """
        Initialize the trainer.

        Args:
            model (torch.nn.Module):
                Model to train.
            modalities (dict[str, str]):
                Modalities trained on.
            num_epochs (int):
                Number of training epochs.
            train_loader (DataLoader):
                Training data loader.
            val_loader (DataLoader):
                Validation data loader.
            weights_loss_seg (float):
                Weight for the segmentation loss.
            classification_weights (dict[str, torch.Tensor]):
                Mapping from classification label name to class weights.
            classification_loss_weights (dict[str, float]):
                Mapping from classification label name to loss weight.
            optimizer (torch.optim.Optimizer):
                Optimizer for training.
            scheduler (torch.optim.lr_scheduler | None):
                Learning rate scheduler.
            device (torch.device):
                Training device.
            config_dir (Config):
                Configuration object.
        """

        self.model = model
        self.modalities = modalities
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weights_loss_seg = weights_loss_seg
        self.classification_weights = classification_weights
        self.classification_loss_weights = classification_loss_weights
        self.classification_tasks = list(classification_weights)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = configIO.Config(config_dir)

    def _train_step(self, epoch: int) -> tuple:
        """
        Run one full training epoch.

        The method computes the task-specific losses, applies the configured
        loss weights, normalizes by the total weight, and updates model
        parameters.

        Args:
            epoch: Current epoch index.

        Returns:
            tuple: Mean training losses for the total objective and each task.
        """

        self.model.to(self.device)

        # Progress bar
        bar_train = tqdm(self.train_loader)

        train_cost = 0.0
        train_cost_seg = 0.0
        train_cost_cls = {
            task_name: 0.0
            for task_name in self.classification_tasks
        }

        # Training mode
        self.model.train(True)

        # Loop over batches and perform training
        for data in bar_train:
            bar_train.set_description(f"Training : Epoch [{epoch+1}/{self.num_epochs}]")

            x = data["image"].to(self.device)

            self.optimizer.zero_grad()

            pred_seg, classification_outputs = self.model(x)

            has_segmentation = pred_seg is not None

            if has_segmentation:
                y_seg = data["mask"].to(self.device)
                loss_seg = dice_loss(pred_seg, y_seg)
                loss_seg_value = loss_seg * self.weights_loss_seg
            else:
                loss_seg_value = torch.tensor(
                    0.0,
                    device=self.device,
                    dtype=x.dtype,
                )

            # Classification losses
            loss_cls_value = {}

            for task_name, prediction in classification_outputs.items():
                target = data[f"label_{task_name}"].to(self.device)
                weights = self.classification_weights[task_name].to(self.device)

                task_loss = masked_cross_entropy_loss(
                    prediction, target, weights
                )
                loss_cls_value[task_name] = (
                    task_loss * self.classification_loss_weights[task_name]
                )

            total_weights = sum(
                self.classification_loss_weights.values()
            )

            if has_segmentation:
                total_weights += self.weights_loss_seg


            loss = loss_seg_value + sum(loss_cls_value.values())
            loss /= total_weights

            loss.backward()
            self.optimizer.step()

            # Normalize for cost logging
            loss_seg_value /= total_weights
            for task_name in loss_cls_value:
                loss_cls_value[task_name] /= total_weights

            train_cost += loss.item()
            train_cost_seg += loss_seg_value.item()

            for task_name in train_cost_cls:
                train_cost_cls[task_name] += loss_cls_value[task_name].item()

        train_cost /= len(self.train_loader)
        train_cost_seg /= len(self.train_loader)

        for task_name in train_cost_cls:
            train_cost_cls[task_name] /= len(self.train_loader)

        return train_cost, train_cost_seg, train_cost_cls

    def _val_step(self, epoch):
        """
        Run one full validation epoch.

        The validation loop mirrors the training loss computation, but it does
        not update model parameters. The same task weights and normalization
        are used so train and validation losses remain comparable.

        Args:
            epoch: Current epoch index.

        Returns:
            tuple: Mean validation losses for the total objective and each
            task.

        """

        self.model.to(self.device)

        # Evaluation mode

        self.model.eval()

        with torch.no_grad():

            bar_val = tqdm(self.val_loader)

            val_cost = 0.0
            val_cost_seg = 0.0
            val_cost_cls = {
                task_name: 0.0
                for task_name in self.classification_tasks
            }

            # Loop over batches and perform validation

            for data in bar_val:

                bar_val.set_description(
                    f"Validation : Epoch [{epoch+1}/{self.num_epochs}]"
                )

                x = data["image"].to(self.device)

                pred_seg, classification_outputs = self.model(x)

                has_segmentation = pred_seg is not None

                if has_segmentation:
                    y_seg = data["mask"].to(self.device)
                    loss_seg = dice_loss(pred_seg, y_seg)
                    loss_seg_value = loss_seg * self.weights_loss_seg
                else:
                    loss_seg_value = torch.tensor(
                        0.0,
                        device=self.device,
                        dtype=x.dtype,
                    )

                # Classification losses
                loss_cls_value = {}

                for task_name, prediction in classification_outputs.items():
                    target = data[f"label_{task_name}"].to(self.device)
                    weights = self.classification_weights[task_name].to(
                        self.device
                    )

                    task_loss = masked_cross_entropy_loss(
                        prediction,
                        target,
                        weights,
                    )

                    loss_cls_value[task_name] = (
                        task_loss
                        * self.classification_loss_weights[task_name]
                    )

                total_weights = sum(
                    self.classification_loss_weights.values()
                )

                if has_segmentation:
                    total_weights += self.weights_loss_seg

                loss = loss_seg_value + sum(loss_cls_value.values())
                loss /= total_weights

                # Normalize for cost logging
                loss_seg_value /= total_weights

                for task_name in loss_cls_value:
                    loss_cls_value[task_name] /= total_weights

                val_cost += loss.item()
                val_cost_seg += loss_seg_value.item()

                for task_name in val_cost_cls:
                    val_cost_cls[task_name] += (
                        loss_cls_value[task_name].item()
                    )

        val_cost /= len(self.val_loader)
        val_cost_seg /= len(self.val_loader)

        for task_name in val_cost_cls:
            val_cost_cls[task_name] /= len(self.val_loader)

        return val_cost, val_cost_seg, val_cost_cls

    def train(
        self,
        experiment_name: str,
        run_name: str,
        experiments_dir: str,
        patience: int,
        delta: float,
        fold_name: str = None,
        run_id: str = None,
    ):
        """
        Train the model and log metrics and artifacts with MLflow.

        The method creates the experiment structure, writes the model summary,
        logs the MLflow signature, runs the epoch loop, and manages early
        stopping and optional scheduler updates.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the run within the experiment.
            experiments_dir: Root directory where experiment results are stored.
            patience: Number of epochs to wait before early stopping.
            delta: Minimum improvement required to reset early stopping.
            fold_name: Optional fold name for cross-validation runs.
            run_id: Optional MLflow run ID for resuming a run.

        Returns:
            None
        """

        if fold_name is not None:
            print("-" * 70)
            print(
                f"Training model for experiment {experiment_name}, and {fold_name}..."
            )
            print("-" * 70)

        else:
            print("-" * 70)
            print(f"Training model for experiment {experiment_name} and run {run_name}")
            print("-" * 70)

        experiments_dir = os.path.join(experiments_dir, experiment_name)
        os.makedirs(experiments_dir, exist_ok=True)

        run_dir = os.path.join(experiments_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        if fold_name is not None:
            fold_dir = os.path.join(run_dir, fold_name)
            os.makedirs(fold_dir, exist_ok=True)
            results_dir = os.path.join(fold_dir, constants.RESULTS_DIR_NAME)
        else:
            results_dir = os.path.join(run_dir, constants.RESULTS_DIR_NAME)

        os.makedirs(results_dir, exist_ok=True)
        model_dir = os.path.join(results_dir, constants.MODELS_DIR_NAME)
        os.makedirs(model_dir, exist_ok=True)

        torch.cuda.empty_cache()

        start_time = timer()

        mlflow.set_experiment(experiment_name)

        os.environ.pop("MLFLOW_RUN_ID", None)

        with mlflow.start_run(run_id=run_id, log_system_metrics=True):

            if fold_name is not None:
                with mlflow.start_run(run_name=fold_name, nested=True):
                    self._train_model_logic(
                        patience,
                        delta,
                        model_dir,
                        start_time,
                    )
            else:
                self._train_model_logic(
                    patience,
                    delta,
                    model_dir,
                    start_time,
                )

    def _train_model_logic(
        self,
        patience,
        delta,
        model_dir,
        start_time,
    ):
        """
        Execute the epoch loop, checkpointing, and metric logging.

        Args:
            patience: Early-stopping patience.
            delta: Early-stopping improvement threshold.
            model_dir: Directory where checkpoints are written.
            start_time: Timestamp marking the start of training.
        """

        early_stopping = EarlyStopping(
            patience=patience, delta=delta, root_path=model_dir
        )
        train_loss = []
        train_loss_seg = []
        train_loss_cls = {
            task_name: []
            for task_name in self.classification_tasks
        }

        val_loss = []
        val_loss_seg = []
        val_loss_cls = {
            task_name: []
            for task_name in self.classification_tasks
        }

        params = {
            "epochs": self.num_epochs,
            "dropout_rate": self.config.train_dropout_rate,
            "learning_rate": self.config.train_optimizer_lr,
            "weight_decay": self.config.train_optimizer_weight_decay,
            "early_stopping_delta": self.config.train_early_stopping_delta,
            "early_stopping_patience": self.config.train_early_stopping_patience,
            "batch_size": self.train_loader.batch_size,
            "loss_function": "Masked dice loss, Masked cross entropy loss",
            "loss_seg": "Segmentation loss: masked dice loss",
            "optimizer": "Adam",
        }

        for task_name in self.classification_tasks:
            params[f"loss_{task_name}"] = (
                f"{task_name} loss: masked cross entropy loss"
            )

        mlflow.log_params(params)

        model_summary = os.path.join(model_dir, constants.MODEL_SUMMARY_FILE)

        with open(model_summary, "w") as f:
            f.write(str(summary(self.model)))

        mlflow.log_artifact(model_summary)

        in_channels = len(self.modalities)

        input_schema = Schema(
            [
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, in_channels, 152, 182, 145),
                    f"MRI scan ({in_channels} modalities)",
                )
            ]
        )
        output_specs = []

        if not getattr(self.model, "classification_only", True):
            output_specs.append(
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, 2, 152, 182, 145),
                    "Tumor segmentation",
                )
            )

        for task_name, num_classes in self.model.classification_tasks.items():
            output_specs.append(
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, num_classes),
                    task_name,
                )
            )

        output_schema = Schema(output_specs)
        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
        )

        for epoch in range(self.num_epochs):

            print(
                f"Epoch {epoch+1} / {self.num_epochs}",
                flush=True,
            )

            (
                train_cost,
                train_cost_seg,
                train_cost_cls,
            ) = self._train_step(epoch=epoch)

            mlflow.pytorch.log_model(
                self.model,
                "model",
                signature=signature,
            )

            (
                val_cost,
                val_cost_seg,
                val_cost_cls,
            ) = self._val_step(epoch=epoch)

            if self.scheduler is not None:

                last_lr = self.scheduler._last_lr[0]

                self.scheduler.step(val_cost)

                new_lr = [
                    group["lr"]
                    for group in self.optimizer.param_groups
                ][0]

                print(f"Scheduler last learning rate: {last_lr}")
                print(f"Current learning rate: {new_lr}")

                if new_lr != last_lr:
                    print(
                        f"Learning rate changed to {new_lr} at epoch {epoch+1}",
                        flush=True,
                    )

            metrics = {
                "Train loss": train_cost,
                "Train loss segmentation": train_cost_seg,
                "Validation loss": val_cost,
                "Validation loss segmentation": val_cost_seg,
            }

            for task_name in self.classification_tasks:
                metrics[f"Train loss {task_name}"] = (
                    train_cost_cls[task_name]
                )
                metrics[f"Validation loss {task_name}"] = (
                    val_cost_cls[task_name]
                )

            mlflow.log_metrics(metrics, step=epoch)

            train_loss.append(train_cost)
            train_loss_seg.append(train_cost_seg)
            val_loss.append(val_cost)
            val_loss_seg.append(val_cost_seg)

            for task_name in self.classification_tasks:
                train_loss_cls[task_name].append(
                    train_cost_cls[task_name]
                )
                val_loss_cls[task_name].append(
                    val_cost_cls[task_name]
                )

            print(
                f"Train loss: {train_cost} | Val loss: {val_cost}",
                flush=True,
            )

            early_stopping(
                current_train_loss=train_cost,
                current_val_loss=val_cost,
                train_loss_history=train_loss,
                train_loss_seg_history=train_loss_seg,
                train_loss_cls_history=train_loss_cls,
                val_loss_history=val_loss,
                val_loss_seg_history=val_loss_seg,
                val_loss_cls_history=val_loss_cls,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
            )

            if self.scheduler is not None:

                if (
                    early_stopping.early_stop
                    and self.optimizer.param_groups[0]["lr"]
                    == self.config.train_optimizer_scheduler_minimum_lr
                ):

                    print(
                        f"Patience of {patience} epochs has been reached and "
                        f"the learning rate is the minimum of "
                        f"{self.config.train_optimizer_scheduler_minimum_lr} : "
                        f"stopping the training.."
                    )

                    early_stopping.save_checkpoint(
                        train_loss=train_loss[-1],
                        val_loss=val_loss[-1],
                        train_loss_history=train_loss,
                        train_loss_seg_history=train_loss_seg,
                        train_loss_cls_history=train_loss_cls,
                        val_loss_history=val_loss,
                        val_loss_seg_history=val_loss_seg,
                        val_loss_cls_history=val_loss_cls,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        file_name="last_model.pt",
                    )
                    break

                elif early_stopping.early_stop:
                    print(
                        f"Patience of {patience} epochs has been reached but "
                        f"the learning rate is not the minimum of "
                        f"{self.config.train_optimizer_scheduler_minimum_lr} : "
                        f"the learning rate will be reduced by the scheduler "
                        f"in the next epoch"
                    )
                    early_stopping.counter = 0
                    early_stopping.early_stop = False

            else:
                if early_stopping.early_stop:
                    print(
                        f"Patience of {patience} epochs has been reached : "
                        f"stopping the training.."
                    )

                    early_stopping.save_checkpoint(
                        train_loss=train_loss[-1],
                        val_loss=val_loss[-1],
                        train_loss_history=train_loss,
                        train_loss_seg_history=train_loss_seg,
                        train_loss_cls_history=train_loss_cls,
                        val_loss_history=val_loss,
                        val_loss_seg_history=val_loss_seg,
                        val_loss_cls_history=val_loss_cls,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        file_name="last_model.pt",
                    )
                    break

        end_time = timer()
        total_time = end_time - start_time

        print(
            f"Total training time : {total_time // 60} minutes, "
            f"{total_time % 60:.4f} seconds"
        )

        print("Finished training..")
        print(
            f"Total training time : {total_time // 60} minutes, "
            f"{total_time % 60:.4f} seconds"
        )

        early_stopping.save_checkpoint(
            train_loss=train_loss[-1],
            val_loss=val_loss[-1],
            train_loss_history=train_loss,
            train_loss_seg_history=train_loss_seg,
            train_loss_cls_history=train_loss_cls,
            val_loss_history=val_loss,
            val_loss_seg_history=val_loss_seg,
            val_loss_cls_history=val_loss_cls,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            file_name="last_model.pt",
        )
