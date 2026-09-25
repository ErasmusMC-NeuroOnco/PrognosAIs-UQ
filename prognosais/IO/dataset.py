import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from prognosais.IO import constants


class DataGenerator:
    """
    Class for loading prognosais dataset (in the format of files path).

    Attributes:
        data_dir : str
            Directory where the data is stored. It should contain a .txt file with the class labels.
        file_extension : str
            Extension of the files to be loaded: should be either '.nii', '.nii.gz', '.dcm'.
        labels_file_dir: str
            String containing the directory of the labels file. It assumes the file has '.txt' extension. If no labels file provided, it
            only loads the image.
        data_type: str
            String containing the type of data to load. Possibilities: "raw", "registered", "preprocessed".
        train : bool
            Boolean variable defining whether to load a train dataset or not.
        subset : float
            Variable defining wether to load the entire dataset (None), or a portion of it ( 0<subset<=1). Default: None.
        missing_value : int
            Integer value representing a missing value within the dataset lables.
        ids_to_exclude: List [str]
            List containing the ids of the patients to exclude from the dataset. Default: None.
        require_mask: bool
            Whether a segmentation mask is required for each patient.
        seed: int
            Random seed used for reproducible subset selection.
        modalities: dict[str, str]
            Mapping of internal modality keys to filenames without extensions.
        structural_modalities: dict[str, str]
            Enabled conventional MRI modalities in canonical channel order.
        mask_file_name: str
            Segmentation mask filename without extension.
        data_folders: dict[str, str]
            Mapping of data-type keys to patient subfolder names.
        labels_config: dict[str, str]
            Mapping of internal classification task keys to labels-file column
            names.
        case_id_column: str
            Labels-file column containing patient identifiers.
        classification_tasks: tuple[str, ...]
            Enabled classification tasks in canonical task order.
        class_values: dict[str, np.ndarray]
            Allowed class values for each classification task.
        label_encoders: dict[str, OneHotEncoder]
            Fitted one-hot encoder for every classification task.
        one_hot_labels: dict[str, torch.Tensor]
            One-hot encoded labels for every classification task.
        class_weights: dict[str, torch.Tensor]
            Balanced class weights for every classification task.
        classification_loss_weights: dict[str, float]
            Task-level loss weights based on label availability.
        weight_loss_seg: float
            Weight assigned to the segmentation loss.
        scans_id: list[str]
            Patient identifiers included in the final dataset.
        labels: pd.DataFrame
            Labels table restricted to patients included in the final dataset.
        original_labels: pd.DataFrame
            Labels table before subset selection and incomplete-patient
            filtering.
        multilabel: pd.DataFrame
            Labels table containing an additional combined-label column.
        included_patient_indices: list[int]
            Indices of patients retained after checking required files.
        data: list[dict]
            MONAI-compatible patient dictionaries. Every dictionary includes
            an explicit case_id entry so downstream inference code does not
            need to recover patient identity from image metadata.

    """

    def __init__(
        self,
        data_dir: str,
        file_extension: str,
        labels_file_dir: Optional[str],
        data_type: str,
        train: bool,
        subset: Union[int, float],
        missing_value: int,
        seed: int,
        modalities: dict[str, str],
        mask_file_name: str,
        data_folders: dict[str, str],
        labels_config: dict[str, str],
        case_id_column: str,
        ids_to_exclude: Optional[list[str]] = None,
        require_mask: bool = False,
    ):

        # Define the class attributes
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.labels_file_dir = labels_file_dir
        self.data_type = data_type
        self.train = train
        self.subset = subset
        self.missing_value = missing_value
        self.seed = seed
        self.modalities = modalities
        self.mask_file_name = mask_file_name
        self.data_folders = data_folders
        self.labels_config = labels_config
        self.case_id_column = case_id_column
        self.classification_tasks = tuple(
            task_name
            for task_name in constants.SUPPORTED_CLASSIFICATION_TASKS
            if task_name in self.labels_config
        )
        self.require_mask = require_mask

        self.structural_modalities = {
            modality_key: self.modalities[modality_key]
            for modality_key in constants.SCAN_TYPES
            if modality_key in self.modalities
        }

        self.class_values = {
            task_name: np.asarray(
                constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                    constants.CLASS_VALUES_KEY
                ]
            )
            for task_name in self.classification_tasks
        }

        self.label_encoders: dict[str, OneHotEncoder] = {}
        self.one_hot_labels: dict[str, torch.Tensor] = {}
        self.class_weights: dict[str, torch.Tensor] = {}
        self.classification_loss_weights: dict[str, float] = {}
        self.weight_loss_seg = 1.0

        files_available = sorted(
            file_name
            for file_name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, file_name))
        )

        if self.labels_file_dir is not None:
            raw_labels = pd.read_csv(
                self.labels_file_dir,
                delimiter="\t",
            )

            if ids_to_exclude is not None:
                raw_labels = raw_labels[
                    ~raw_labels[self.case_id_column].isin(ids_to_exclude)
                ]

            labels_available = raw_labels[self.case_id_column].astype(str)

            self.scans_id = sorted(set(labels_available) & set(files_available))

            label_mask = raw_labels[self.case_id_column].astype(str).isin(self.scans_id)

            self.labels = raw_labels.loc[label_mask].copy()

            self.labels[self.case_id_column] = self.labels[self.case_id_column].astype(
                str
            )

            self.labels = self.labels.sort_values(
                by=self.case_id_column,
                ignore_index=True,
            )

            self.original_labels = self.labels.copy(deep=True)

            self.features = [
                self.labels_config[task_name] for task_name in self.classification_tasks
            ]

            self.multilabel = self.labels.copy(deep=True)
            combinations = self.multilabel[self.features].apply(
                tuple,
                axis=1,
            )
            self.multilabel["combined_label"] = pd.factorize(combinations)[0]

            if self.subset != 1:
                num_elements = max(
                    1,
                    int(len(self.scans_id) * self.subset),
                )

                rng = np.random.default_rng(self.seed)
                self.indices = sorted(
                    rng.choice(
                        len(self.scans_id),
                        size=num_elements,
                        replace=False,
                    ).tolist()
                )

                self.scans_id = [self.scans_id[index] for index in self.indices]

                self.labels = self.labels.iloc[self.indices].sort_values(
                    by=self.case_id_column,
                    ignore_index=True,
                )

                self.multilabel = self.multilabel.iloc[self.indices].sort_values(
                    by=self.case_id_column,
                    ignore_index=True,
                )

            self.one_hot_labels = self._get_one_hot_labels()

        else:
            self.scans_id = files_available

            if self.subset != 1:
                num_elements = max(
                    1,
                    int(len(self.scans_id) * self.subset),
                )

                rng = np.random.default_rng(self.seed)
                self.indices = sorted(
                    rng.choice(
                        len(self.scans_id),
                        size=num_elements,
                        replace=False,
                    ).tolist()
                )

                self.scans_id = [self.scans_id[index] for index in self.indices]

        self.data = self._get_data()

        if self.labels_file_dir is not None:
            included_indices = self.included_patient_indices

            self.scans_id = [self.scans_id[index] for index in included_indices]

            self.labels = self.labels.iloc[included_indices].reset_index(drop=True)

            self.multilabel = self.multilabel.iloc[included_indices].reset_index(
                drop=True
            )

            self.one_hot_labels = {
                task_name: one_hot[included_indices]
                for task_name, one_hot in self.one_hot_labels.items()
            }

            if self.train:
                self.class_weights = self._get_class_weights()
                self.classification_loss_weights = (
                    self._get_classification_loss_weights()
                )

                print(f"Segmentation loss weight: " f"{self.weight_loss_seg}")

                for task_name in self.classification_tasks:
                    print(
                        f"Class weights for {task_name}: "
                        f"{self.class_weights[task_name]}"
                    )
                    print(
                        f"Loss weight for {task_name}: "
                        f"{self.classification_loss_weights[task_name]}"
                    )

    def __len__(self):
        """
        Retrieves the length of the dataset

        Args:
            dataset object

        Returns:
            int
                Dataset length

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Function to obtain one data sample from the dataset.

        Args:
            idx: int
                Index of the data sample to retrieve.

        Returns:
            data[idx]: Dir[str, str]
                Dictionary containing the data type and file path for the given index.
        """
        return self.data[idx]

    def _get_data(self) -> list[dict]:
        """
        Build one MONAI input dictionary per complete patient. Patients missing a configured modality or required mask are skipped.

        Returns:
            list[dict]:
                Patient dictionaries containing scan paths, optional mask paths,
                and optional classification labels.
        """
        data = []
        skipped_patients = []
        included_patient_indices = []

        for patient_index, patient_id in enumerate(self.scans_id):
            scans_dir = os.path.join(
                self.data_dir,
                patient_id,
                self.data_folders[self.data_type],
            )

            if not os.path.isdir(scans_dir):
                skipped_patients.append((patient_id, f"missing directory: {scans_dir}"))
                continue

            structural_paths = [
                os.path.join(
                    scans_dir,
                    modality_filename + self.file_extension,
                )
                for modality_filename in self.structural_modalities.values()
            ]

            missing_files = [
                path
                for path in structural_paths
                if not os.path.isfile(path)
            ]

            mask_path = None
            if self.require_mask:
                mask_path = os.path.join(
                    scans_dir,
                    self.mask_file_name + self.file_extension,
                )

                if not os.path.isfile(mask_path):
                    missing_files.append(mask_path)

            if missing_files:
                skipped_patients.append(
                    (
                        patient_id,
                        "missing files: " + ", ".join(missing_files),
                    )
                )
                continue

            data_dict = {
                constants.CASE_ID_DATA_KEY: patient_id,
            }

            if structural_paths:
                data_dict["structural"] = structural_paths

            if self.require_mask:
                data_dict["mask"] = mask_path

            if self.labels_file_dir is not None:
                for task_name in self.classification_tasks:
                    data_dict[f"label_{task_name}"] = self.one_hot_labels[task_name][
                        patient_index
                    ]

            data.append(data_dict)
            included_patient_indices.append(patient_index)

        if skipped_patients:
            print(f"Skipped {len(skipped_patients)} incomplete patients:")
            for patient_id, reason in skipped_patients:
                print(f"  - {patient_id}: {reason}")

        self.included_patient_indices = included_patient_indices
        return data

    def _get_one_hot_labels(
        self,
    ) -> dict[str, torch.Tensor]:
        """
        One-hot encode all configured classification tasks.

        Returns:
            dict[str, torch.Tensor]:
                Mapping of classification task keys to one-hot label tensors.
        """
        one_hot_labels = {}

        for task_name in self.classification_tasks:
            column_name = self.labels_config[task_name]
            class_values = self.class_values[task_name]

            labels = self.labels[column_name].fillna(self.missing_value).to_numpy()

            encoder = OneHotEncoder(
                categories=[class_values],
                handle_unknown="ignore",
                sparse_output=False,
            )

            encoder.fit(class_values.reshape(-1, 1))

            encoded_labels = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)

            one_hot_labels[task_name] = torch.from_numpy(encoded_labels)

            self.label_encoders[task_name] = encoder

        return one_hot_labels

    def _get_class_weights(
        self,
    ) -> dict[str, torch.Tensor]:
        """
        Compute balanced class weights for each configured task.

        Returns:
            dict[str, torch.Tensor]:
                Mapping of classification task keys to class-weight tensors.
        """
        class_weights = {}

        for task_name in self.classification_tasks:
            one_hot = self.one_hot_labels[task_name]

            known_labels = one_hot[one_hot.sum(dim=1) > 0].argmax(dim=1)

            present_classes = np.unique(known_labels.numpy())
            expected_classes = np.arange(constants.NUM_CLASSES_PER_LABEL[task_name])

            if not np.array_equal(present_classes, expected_classes):
                raise ValueError(
                    f"Task '{task_name}' is missing training classes. "
                    f"Expected {expected_classes.tolist()}, "
                    f"found {present_classes.tolist()}."
                )

            weights = compute_class_weight(
                class_weight="balanced",
                classes=expected_classes,
                y=known_labels.numpy(),
            )

            class_weights[task_name] = torch.from_numpy(weights).to(torch.float32)

        return class_weights

    def _get_classification_loss_weights(
        self,
    ) -> dict[str, float]:
        """
        Compute task-level loss weights based on label availability.

        Returns:
            dict[str, float]:
                Mapping of classification task keys to task-level loss weights.
        """
        labels_total = len(self.labels)
        classification_loss_weights = {}

        for task_name in self.classification_tasks:
            column_name = self.labels_config[task_name]

            known_mask = self.labels[column_name].notna() & (
                self.labels[column_name] != self.missing_value
            )

            number_known = int(known_mask.sum())

            if number_known == 0:
                raise ValueError(
                    f"No known labels are available for task '{task_name}'."
                )

            classification_loss_weights[task_name] = round(
                labels_total / number_known,
                2,
            )

        return classification_loss_weights


class KFoldGenerator:
    """

    Class for generating the indices for Kfold cross validation.

    Attributes:


    Methods:
        __init__(self, num_folds)
            Initializes the class attributes.
        get_folds_stratified(self, X, y)
            Get the indices of data folds in a stratified fashion.
        get_folds(self, X)
            Get the indices of data folds in a random fashion.

    """

    def __init__(self, num_folds, seed: int):
        """
        Function to initialize .

        Args:
            self
                Internal-use class function.

        Returns:
            weight_idh, weight_1p19q, weight_grade: torch.Tensor
                Tensor containing the one-hot encoded labels for its respective feature.

        """
        super().__init__()
        self.num_folds = num_folds
        self.seed = seed

    def get_folds_stratified(self, X, y):
        """
        Function to obtain the indices for cross validation in a stratified fashion.

        Args:
            X : array-like object
                Data to split
            y : array-like object
                Target value (labels)

        Returns:
            idx : List[List[int], List[int]]
                List with indices for training and validation in each fold.

        """
        # Generate the folds indices and save them in a list.
        kf = StratifiedKFold(
            n_splits=self.num_folds, shuffle=True, random_state=self.seed
        )
        idx = list(kf.split(X=X, y=y))
        return idx

    def get_folds(self, X):
        """
        Function to obtain the data indices for cross validation.

        Args:
            X : dataset to split
        Returns:
            idx : List[List[int], List[int]]
                List with indices for training and validation in each fold.

        """
        # Generate the folds indices and save them in a list.
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        idx = list(kf.split(X=X))
        return idx
