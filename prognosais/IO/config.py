from pathlib import Path
from typing import Any, Tuple, Union

import yaml

import prognosais.IO.constants as constants


class Config:
    """
    Config class for managing configuration settings.
    This class provides methods and properties to load, validate, and access
    configuration settings from a YAML file. It ensures that the configuration
    values are of the correct type and provides detailed error handling for
    missing or invalid configuration entries.
    Attributes:
        config_file (str): Path to the YAML configuration file.
        config (dict): Dictionary containing the loaded configuration data.
    Methods:
        load_config:
            Loads the configuration from the specified YAML file.
        ensure_config_value_has_correct_type:
            Validates that a configuration value matches the expected type.
    Properties:
        data_train_dir (str):
            Directory path for training data.
        data_train_subset (float):
            Subset of training data specified in the configuration.
        data_train_type (str):
            Type of training data.
        data_train_missing_value (int):
            Missing value for training data.
        data_test_dir (str):
            Directory path for test data.
        data_test_subset (float):
            Subset of test data specified in the configuration.
        data_test_type (str):
            Type of test data.
        data_test_missing_value (int):
            Missing value for test data.
        data_preprocess_dir (str):
            Directory path for preprocessing data.
        preprocess_DICOM_data (bool):
            Whether to preprocess DICOM data.
        preprocess_registered_data (bool):
            Whether scans/mask are already registered.
        bias_field_correct_data (bool):
            Whether scans need bias field correction.
        preprocess_registered_data (bool):
            Whether scans/mask are already registered.
        bias_field_correct_data (bool):
            Whether scans need bias field correction.
        slurm_cluster (str):
            SLURM cluster configuration value.
        cpu_node_partition (str):
            Partition name for the CPU node in SLURM.
        cpu_node_job_header (str):
            Job header configuration for a CPU node in SLURM.
        gpu_node_partition (str):
            Partition name for the GPU node in SLURM.
        gpu_node_job_header (str):
            Job header configuration for a GPU node in SLURM.
        gpu_node_gres (dict):
            GPU node GRES (Generic Resource) configuration in SLURM.
        slurm_email (str):
            Email address for SLURM notifications.
        slurm_train_time (str):
            SLURM training time configuration.
        slurm_test_time (str):
            SLURM test time configuration.
        slurm_preprocess_time (str):
            SLURM preprocess time configuration.
        slurm_preprocess_time (str):
            SLURM preprocess time configuration.
        experiment_name (str):
            Experiment name for organizing outputs.
        run_name (str):
            Name for this model run.
        experiments_dir (str):
            Parent directory for experiment outputs.
        container_path (Path):
            Directory path where container images are stored.
        hdbet_version (str):
            HDBET container version string.
        fsl_version (str):
            FSL container version string.
        environment_seed (int):
            Seed value for the environment.
        train_kfold (bool):
            Whether k-fold cross-validation is enabled during training.
        train_num_folds (int):
            Number of folds for training.
        train_epochs (int):
            Number of training epochs.
        train_batch_size (int):
            Batch size for training.
        train_dropout_rate (float):
            Dropout rate for training.
        train_optimizer_lr (float):
            Learning rate for the optimizer.
        train_optimizer_weight_decay (float):
            Weight decay value for the optimizer.
        train_early_stopping_delta (Union[int, float]):
            Early stopping delta value.
        train_early_stopping_patience (int):
            Early stopping patience value.
        test_kfold (bool):
            Whether k-fold cross-validation is enabled during testing.
        test_batch_size (int):
            Batch size for testing.
        test_num_folds (int):
            Number of folds for testing.
        test_results_dir (str):
            Directory path for test results.
        test_model_dir (str):
            Directory path of the model to use for testing.
        test_save_predictions (bool):
            Whether to save predictions during testing.
        mc_dropout_samples (int):
            Number of Monte Carlo dropout samples.
        dropout_experiment_name (str):
            Experiment name for the MC Dropout experiment.
    """

    def __init__(self, config_file: str) -> None:
        """Initialize the configuration wrapper.

        Args:
            config_file: Path to the YAML configuration file that should be
                loaded and validated.
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads the configuration from a YAML file.
        This method reads the YAML configuration file specified by self.config_file
        and returns its contents as a dictionary.
        Returns:
            dict: The configuration data loaded from the YAML file.
        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """

        with open(self.config_file, "r") as the_config_file:
            config = yaml.load(the_config_file, Loader=yaml.SafeLoader)
        return config

    def ensure_config_value_has_correct_type(
        self,
        config_value: Union[str, int, float, bool, dict, list],
        expected_type: Union[type, Tuple[Any, ...]],
    ) -> None:
        """
        Ensures that a configuration value has the expected type.
        This method checks if the provided configuration value matches the expected type.
        If the type does not match, it raises a TypeError with a descriptive message.
        Args:
            config_value (Union[str, int, float, bool, dict]):
                The configuration value to check.
            expected_type (Union[str, int, float, bool, dict] | Tuple[
                Union[str, int, float, bool, dict], Union[str, int, float, bool, dict]
            ]):
                The expected type of the configuration value.
        Raises:
            TypeError: If the configuration value does not match the expected type.
        """

        if type(expected_type) is tuple:
            if type(config_value) not in expected_type:
                raise TypeError(
                    f"Expected {[expect_name.__name__ for expect_name in expected_type]}, got {type(config_value).__name__}"
                )
        else:
            if type(config_value) is not expected_type:
                raise TypeError(
                    f"Expected {expected_type.__name__}, got {type(config_value).__name__}"
                )

    @property
    def data_modalities(self) -> dict[str, str]:
        """
        Mapping of conventional structural MRI modalities to use for training
        and testing (T1, T1CE, T2, and FLAIR).

        Returns:
            dict: dict of MRI modalities.
        """
        configured_modalities = self.config["data"]["modalities"]
        self.ensure_config_value_has_correct_type(configured_modalities, dict)

        enabled_modalities = {}

        for modality_key, modality_filename in configured_modalities.items():
            self.ensure_config_value_has_correct_type(modality_key, str)
            self.ensure_config_value_has_correct_type(modality_filename, str)

            if not modality_filename.strip():
                continue

            if modality_key not in constants.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unknown modality '{modality_key}'. "
                    f"Supported modalities are: "
                    f"{sorted(constants.SUPPORTED_MODALITIES)}"
                )

            enabled_modalities[modality_key] = modality_filename

        return enabled_modalities

    @property
    def data_mask_file_name(self) -> str:
        """
        Retrieve the filename used for the segmentation mask annotation.

        The returned value is the file name expected inside each patient
        directory for mask-based preprocessing and training tasks.

        Returns:
            str: The configured mask filename.
        """
        mask_file_name = self.config["data"]["mask_file_name"]
        self.ensure_config_value_has_correct_type(mask_file_name, str)
        return mask_file_name

    @property
    def data_folders(self) -> dict:
        """
        Retrieve the configured data folder mapping.

        The configuration is expected to define the folder names for the
        original, registered, and preprocessed data layouts. These values are
        used throughout preprocessing and dataset construction.

        Returns:
            dict: Mapping of data stage names to folder names.
        """
        folders = self.config["data"]["folders"]
        self.ensure_config_value_has_correct_type(folders, dict)
        for data_type in ("original", "registered", "preprocessed"):
            if data_type not in folders:
                raise KeyError(f"Missing data folder mapping for '{data_type}'")
            self.ensure_config_value_has_correct_type(folders[data_type], str)
        return folders

    @property
    def data_original_folder(self) -> str:
        """
        Retrieve the folder name for the original data layout.

        This is the subfolder name used for the raw input data before any
        preprocessing or registration has been applied.

        Returns:
            str: Folder name for original data.
        """
        folder = self.data_folders["original"]
        self.ensure_config_value_has_correct_type(folder, str)
        return folder

    @property
    def data_registered_folder(self) -> str:
        """
        Retrieve the folder name for registered data.

        This folder name is used when scans and masks have already been brought
        into a common reference space.

        Returns:
            str: Folder name for registered data.
        """
        folder = self.data_folders["registered"]
        self.ensure_config_value_has_correct_type(folder, str)
        return folder

    @property
    def data_preprocessed_folder(self) -> str:
        """
        Retrieve the folder name for preprocessed data.

        This is the output folder name used by preprocessing steps that prepare
        the inputs for training or inference.

        Returns:
            str: Folder name for preprocessed data.
        """
        folder = self.data_folders["preprocessed"]
        self.ensure_config_value_has_correct_type(folder, str)
        return folder

    @property
    def data_labels(self) -> dict:
        """
        Return and validate the label configuration block.
        """
        labels = self.config["data"]["labels"]
        self.ensure_config_value_has_correct_type(labels, dict)

        case_id_column = labels.get("case_id_column")
        self.ensure_config_value_has_correct_type(case_id_column, str)

        tasks = labels.get("tasks")
        self.ensure_config_value_has_correct_type(tasks, dict)

        for task_name, column_name in tasks.items():
            self.ensure_config_value_has_correct_type(task_name, str)

            if column_name is None:
                continue

            self.ensure_config_value_has_correct_type(column_name, str)

            if task_name not in constants.CLASSIFICATION_TASK_DEFINITIONS:
                valid_tasks = sorted(constants.CLASSIFICATION_TASK_DEFINITIONS)
                raise ValueError(
                    f"Unknown classification task '{task_name}'. "
                    f"Valid tasks are: {valid_tasks}"
                )

        return labels

    @property
    def data_label_case_id_column(self) -> str:
        """
        Return the column containing patient or case identifiers.
        """
        return self.data_labels["case_id_column"]

    @property
    def labels(self) -> dict[str, str]:
        """
        Map enabled task names to label-table columns.

        Example:
            {
                "idh": "IDH",
                "onep19q": "1p19q",
                "grade": "Grade",
            }

        """
        configured_tasks = self.data_labels["tasks"]

        enabled_tasks = {
            task_name: column_name.strip()
            for task_name, column_name in configured_tasks.items()
            if isinstance(column_name, str) and column_name.strip()
        }

        if not enabled_tasks:
            raise ValueError(
                "No classification tasks are enabled under " "'data.labels.tasks'."
            )

        return enabled_tasks

    @property
    def classification_tasks(self) -> tuple[str, ...]:
        """
        Return the enabled classification task names.
        """
        return tuple(self.labels)

    def get_label_column(self, task_name: str) -> str:
        """
        Return the labels-file column for an enabled task.
        """
        if task_name not in self.labels:
            raise KeyError(
                f"Classification task '{task_name}' is not enabled. "
                f"Enabled tasks are: {list(self.labels)}"
            )

        return self.labels[task_name]

    def get_label_definition(self, task_name: str) -> dict:
        """
        Return the constant task definition.
        """
        try:
            return constants.CLASSIFICATION_TASK_DEFINITIONS[task_name]
        except KeyError as error:
            valid_tasks = sorted(constants.CLASSIFICATION_TASK_DEFINITIONS)
            raise KeyError(
                f"Unknown classification task '{task_name}'. "
                f"Valid tasks are: {valid_tasks}"
            ) from error

    def get_label_output_key(self, task_name: str) -> str:
        return self.get_label_definition(task_name)[constants.OUTPUT_KEY_KEY]

    def get_label_class_values(
        self,
        task_name: str,
    ) -> tuple[int, ...]:
        return self.get_label_definition(task_name)[constants.CLASS_VALUES_KEY]

    def get_label_class_names(
        self,
        task_name: str,
    ) -> tuple[str, ...]:
        return self.get_label_definition(task_name)[constants.CLASS_NAMES_KEY]

    def get_label_num_classes(self, task_name: str) -> int:
        return self.get_label_definition(task_name)[constants.NUM_CLASSES_KEY]

    @property
    def data_train_dir(self) -> str:
        """
        Get the directory path for training data.

        This property retrieves the directory path where the training data is stored
        from the configuration dictionary. It also ensures that the retrieved value
        is of the correct type (string).

        Returns:
            str: The directory path for training data.

        Raises:
            TypeError: If the retrieved configuration value is not of type string.
        """
        train_dir = self.config["data"]["train"]["input_dir"]
        self.ensure_config_value_has_correct_type(train_dir, str)
        return train_dir

    @property
    def data_train_labels_file_dir(self) -> Union[str, None]:
        """
        Get the directory path for training labels file.

        This property retrieves the directory path where the training labels file is stored
        from the configuration file. It also ensures that the retrieved value
        is of the correct type (string) or None in case it is not provided.

        Returns:
            Union[str, None]: The directory path for training labels file or None if not specified.

        Raises:
            TypeError: If the retrieved configuration value is not of type string or None.
        """
        train_labels_dir = self.config["data"]["train"]["labels_file_path"]
        if train_labels_dir is not None:
            self.ensure_config_value_has_correct_type(train_labels_dir, str)
        return train_labels_dir

    @property
    def data_train_subset(self) -> float:
        """
        Retrieves the subset of training data specified in the configuration.
        This method accesses the configuration dictionary to fetch the subset value
        for training data. It ensures that the value is of the correct type (either
        int or float) before returning it.
        Returns:
            float: The subset of training data as specified in the configuration.
        Raises:
            TypeError: If the subset value is not of type int or float.
        """
        data_subset = self.config["data"]["train"]["subset"]
        self.ensure_config_value_has_correct_type(data_subset, (int, float))

        if not 0 < data_subset <= 1:
            raise ValueError(
                f"Training subset must satisfy 0 < subset <= 1, got {data_subset}."
            )

        return float(data_subset)

    @property
    def data_train_type(self) -> str:
        """
        Retrieves the type of training data from the configuration.
        This method accesses the configuration dictionary to fetch the type of
        training data specified under the "data" -> "train" -> "type" keys. It
        ensures that the retrieved value is of the correct type (string).
        Returns:
            str: The type of training data as specified in the configuration.
        Raises:
            TypeError: If the retrieved configuration value is not of type str.
        """

        data_type = self.config["data"]["train"]["type"]
        self.ensure_config_value_has_correct_type(data_type, str)

        if data_type not in self.data_folders:
            raise ValueError(
                f"Unknown training data type '{data_type}'. "
                f"Supported data types are: {sorted(self.data_folders)}"
            )

        return data_type

    @property
    def data_train_missing_value(self) -> int:
        """
        Retrieves the missing value for training data from the configuration.
        This method accesses the configuration dictionary to fetch the missing
        value specified for the training data. It ensures that the retrieved
        value is of the correct type (int) before returning it.
        Returns:
            int: The missing value for training data.
        Raises:
            TypeError: If the missing value is not of type int.
        """

        missing_value = self.config["data"]["train"]["missing_value"]
        self.ensure_config_value_has_correct_type(missing_value, int)
        return missing_value

    @property
    def data_train_augmentation_factor(self) -> int:
        """
        Retrieves the augmentation factor for training data from the configuration.
        This method accesses the configuration dictionary to fetch the augmentation
        factor specified for the training data. It ensures that the retrieved value
        is of the correct type (int) before returning it.
        Returns:
            int: The augmentation factor for training data.
        Raises:
            TypeError: If the augmentation factor is not of type int.
        """

        augmentation_factor = self.config["data"]["train"]["augmentation_factor"]
        self.ensure_config_value_has_correct_type(augmentation_factor, int)
        return augmentation_factor

    @property
    def data_train_augmentation_probability(self) -> float:
        """
        Retrieves the transformation probability for training data from the configuration.
        This method accesses the configuration dictionary to fetch the transformation
        probability specified for the training data. It ensures that the retrieved value
        is of the correct type (float) before returning it.
        Returns:
            float: The transformation probability for training data.
        Raises:
            TypeError: If the transformation probability is not of type float.
        """

        augmentation_probability = self.config["data"]["train"][
            "augmentation_probability"
        ]
        self.ensure_config_value_has_correct_type(augmentation_probability, float)
        return augmentation_probability

    @property
    def data_test_dir(self) -> str:
        """
        Retrieves the test data directory path from the configuration.
        This method accesses the configuration dictionary to fetch the path
        of the test data directory. It also ensures that the retrieved value
        is of the correct type (string).
        Returns:
            str: The path to the test data directory.
        Raises:
            TypeError: If the retrieved configuration value is not a string.
        """

        test_dir = self.config["data"]["test"]["input_dir"]
        self.ensure_config_value_has_correct_type(test_dir, str)
        return test_dir

    @property
    def data_test_labels_file_dir(self) -> Union[str, None]:
        """
        Get the directory path for test labels file.

        This property retrieves the directory path where the test labels file is stored
        from the configuration file. It also ensures that the retrieved value
        is of the correct type (string) or None in case it is not provided.

        Returns:
            Union[str, None]: The directory path for test labels file or None if not specified.

        Raises:
            TypeError: If the retrieved configuration value is not of type string or None.
        """
        test_labels_dir = self.config["data"]["test"]["labels_file_path"]
        if test_labels_dir is not None:
            self.ensure_config_value_has_correct_type(test_labels_dir, str)
        return test_labels_dir

    @property
    def data_test_inference_mode(self) -> str:
        """
        Retrieve and validate the label mode for every test-data workflow.

        The same test dataset configuration is used by deterministic inference,
        Monte Carlo dropout, deep ensembles, Monte Carlo deep ensembles, and
        segmentation uncertainty aggregation. Labeled mode requires a real
        test-label file. Unlabeled mode requires labels_file_path to be null so
        no workflow can accidentally consume labels from a previous setup.

        Returns:
            str: Either labeled or unlabeled.

        Raises:
            KeyError: If data.test does not define inference_mode.
            TypeError: If inference_mode is not a string.
            ValueError: If inference_mode is unsupported or contradicts
                labels_file_path.
            FileNotFoundError: If labeled mode references a missing labels file.
        """
        test_config = self.config["data"]["test"]
        if "inference_mode" not in test_config:
            raise KeyError("Missing 'inference_mode' under data.test")
        inference_mode = test_config["inference_mode"]
        self.ensure_config_value_has_correct_type(inference_mode, str)
        if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
            supported_modes = ", ".join(constants.SUPPORTED_INFERENCE_MODES)
            raise ValueError(
                "Unsupported data.test.inference_mode "
                f"'{inference_mode}'. Choose one of: {supported_modes}."
            )

        labels_file_path = self.data_test_labels_file_dir
        if inference_mode == constants.INFERENCE_MODE_LABELED:
            if labels_file_path is None or not labels_file_path.strip():
                raise ValueError(
                    "data.test.inference_mode is labeled, but "
                    "data.test.labels_file_path is empty."
                )
            if not Path(labels_file_path).is_file():
                raise FileNotFoundError(
                    "data.test.inference_mode is labeled, but the configured "
                    f"labels file does not exist: {labels_file_path}"
                )
        elif labels_file_path is not None:
            raise ValueError(
                "data.test.inference_mode is unlabeled, so "
                "data.test.labels_file_path must be null."
            )
        return inference_mode

    @property
    def data_test_subset(self) -> float:
        """
        Retrieves the subset value for test data from the configuration.
        This method fetches the subset value for test data from the configuration
        dictionary and ensures that it is of the correct type (either int or float).
        Returns:
            float: The subset value for test data.
        Raises:
            TypeError: If the subset value is not of type int or float.
        """

        data_subset = self.config["data"]["test"]["subset"]
        self.ensure_config_value_has_correct_type(data_subset, (int, float))

        if not 0 < data_subset <= 1:
            raise ValueError(
                f"Training subset must satisfy 0 < subset <= 1, got {data_subset}."
            )

        return float(data_subset)

    @property
    def data_test_type(self) -> str:
        """
        Retrieves the type of test data from the configuration.
        This method accesses the configuration dictionary to fetch the type of test data specified.
        It ensures that the retrieved value is of the correct type (string).
        Returns:
            str: The type of test data as specified in the configuration.
        Raises:
            TypeError: If the retrieved data type is not a string.
        """

        data_type = self.config["data"]["test"]["type"]
        self.ensure_config_value_has_correct_type(data_type, str)

        if data_type not in self.data_folders:
            raise ValueError(
                f"Unknown test data type '{data_type}'. "
                f"Supported data types are: {sorted(self.data_folders)}"
            )

        return data_type

    @property
    def data_test_missing_value(self) -> int:
        """
        Retrieves the missing value for test data from the configuration.
        This method fetches the missing value specified in the configuration
        under the "data" -> "test" -> "missing_value" path and ensures it is
        of the correct type (int).
        Returns:
            int: The missing value for test data.
        Raises:
            TypeError: If the missing value is not of type int.
        """

        missing_value = self.config["data"]["test"]["missing_value"]
        self.ensure_config_value_has_correct_type(missing_value, int)
        return missing_value

    @property
    def data_preprocess_dir(self) -> str:
        """
        Retrieves the directory path for preprocessing data from the configuration.
        This method accesses the configuration dictionary to fetch the path
        where the preprocessing data is stored. It ensures that the retrieved
        value is of the correct type (string).
        Returns:
            str: The directory path for preprocessing data.
        Raises:
            TypeError: If the retrieved configuration value is not a string.
        """
        # Ensure the config value is a string
        preprocess_dir = self.config["data"]["preprocess"]["input_dir"]
        self.ensure_config_value_has_correct_type(preprocess_dir, str)
        return preprocess_dir

    @property
    def mask_origin_file_path(self) -> Union[str, None]:
        """
        Get the directory path for mask_origin file of data to preprocess
        This property retrieves (from the configuration file) the directory path of the mask_origin file for
        data to preprocess. It also ensures that the retrieved value is of the correct type (string) or None
        in case there is no path in the configuration file

        Returns:
            Union[str, None]: The directory path for mask_origin file or None if not specified.
        Raises:
            TypeError: If the retrieved configuration value is not of type string or None.
        """
        preprocess_mask_origin_path = self.config["data"]["preprocess"][
            "mask_origin_file_path"
        ]
        if not preprocess_mask_origin_path:
            return None

        self.ensure_config_value_has_correct_type(preprocess_mask_origin_path, str)
        return preprocess_mask_origin_path

    @property
    def preprocess_DICOM_data(self) -> bool:
        """
        Check if scans are in DICOM and need conversion to NIFTI. Defaults to False.

        Returns:
            bool: Whether data is in DICOM.
        """
        preprocess_DICOM = self.config["data"]["preprocess"].get("data_is_DICOM", False)
        if preprocess_DICOM is not None:
            self.ensure_config_value_has_correct_type(preprocess_DICOM, bool)
            return preprocess_DICOM
        return False

    @property
    def preprocess_registered_data(self) -> bool:
        """
        Check if scans/mask are already registered. Defaults to False.

        Returns:
            bool: Whether scans/mask are already registered.
        """
        preprocess_registered_data = self.config["data"]["preprocess"][
            "already_registered"
        ]
        if preprocess_registered_data is not None:
            self.ensure_config_value_has_correct_type(preprocess_registered_data, bool)
            return preprocess_registered_data
        else:
            return False

    @property
    def bias_field_correct_data(self) -> bool:
        """
        Check if scans need bias field correction. Defaults to true.

        Returns:
            bool: Whether scans need bias field correction.
        """
        bias_field_correct = self.config["data"]["preprocess"]["bias_field_correction"]
        if bias_field_correct is not None:
            self.ensure_config_value_has_correct_type(bias_field_correct, bool)
            return bias_field_correct
        else:
            return True

    @property
    def slurm_cluster(self) -> str:
        """
        Retrieves the SLURM cluster configuration value.
        This method fetches the SLURM cluster configuration value from the
        configuration dictionary and ensures that it is of the correct type.
        Returns:
            str: The SLURM cluster configuration value.
        Raises:
            TypeError: If the configuration value is not of type str.
        """

        cluster = self.config["slurm"]["cluster"]
        self.ensure_config_value_has_correct_type(cluster, str)
        return cluster

    @property
    def slurm_cluster_username(self) -> str:
        """
        Retrieves the SLURM cluster username configuration value.
        This method fetches the SLURM cluster username from the configuration
        dictionary and ensures that it is of the correct type (string).
        Returns:
            str: The SLURM cluster username.
        Raises:
            TypeError: If the retrieved value is not of type string.
        """

        cluster_username = self.config["slurm"]["cluster_username"]
        self.ensure_config_value_has_correct_type(cluster_username, str)
        return cluster_username

    @property
    def cpu_node_partition(self) -> str:
        """
        Retrieves the partition name for the CPU node from the SLURM configuration.
        This method accesses the SLURM configuration dictionary to fetch the partition
        name for the CPU node. It also ensures that the retrieved value is of the correct type.
        Returns:
            str: The partition name for the CPU node.
        Raises:
            TypeError: If the partition value is not of type str.
        """

        partition = self.config["slurm"]["cpu_node"]["partition"]
        self.ensure_config_value_has_correct_type(partition, str)
        return partition

    @property
    def cpu_node_job_header(self) -> str:
        """
        Retrieves the job header configuration for a CPU node.
        This method fetches the job header configuration for a CPU node from the
        SLURM configuration and ensures that the value is of the correct type (str).
        Returns:
            str: The job header configuration for a CPU node.
        Raises:
            TypeError: If the job header configuration is not of type str.
        """

        job_header = self.config["slurm"]["cpu_node"]["job_header"]
        self.ensure_config_value_has_correct_type(job_header, str)
        return job_header

    @property
    def cpu_node_cpus_per_task(self) -> int:
        """
        Retrieve the CPU count requested for each CPU-node Slurm task.

        The CPU-node resource request is defined in the configuration so CPU
        aggregation jobs can be adapted to a cluster's allocation policy without
        editing pipeline code.

        Returns:
            int: Positive number of CPUs requested for one Slurm task.

        Raises:
            TypeError: If the configured value is not an integer.
            ValueError: If the configured CPU count is not positive.
        """

        cpus_per_task = self.config["slurm"]["cpu_node"]["cpus_per_task"]
        self.ensure_config_value_has_correct_type(cpus_per_task, int)
        if cpus_per_task <= 0:
            raise ValueError("slurm.cpu_node.cpus_per_task must be positive.")
        return cpus_per_task

    def cpu_node_memory_size(self, job_key: str) -> str:
        """
        Retrieve the CPU-node memory request for one named job type.

        Memory requests are configured separately by job type because deep
        ensemble aggregation, MC Dropout aggregation, and mask-based UQ
        aggregation have different peak memory requirements.

        Args:
            job_key: Key under slurm.cpu_node.memory that identifies the job
                type requesting memory.

        Returns:
            str: Slurm-compatible memory request, for example 400GB.

        Raises:
            TypeError: If job_key or the configured memory value is not a string,
                or if the memory mapping is not a dictionary.
            KeyError: If the configuration does not define memory for job_key.
            ValueError: If the configured memory value is empty.
        """

        self.ensure_config_value_has_correct_type(job_key, str)
        memory_by_job = self.config["slurm"]["cpu_node"]["memory"]
        self.ensure_config_value_has_correct_type(memory_by_job, dict)
        if job_key not in memory_by_job:
            raise KeyError(
                "Missing CPU-node memory configuration for job type "
                f"'{job_key}' under slurm.cpu_node.memory."
            )
        memory_size = memory_by_job[job_key]
        self.ensure_config_value_has_correct_type(memory_size, str)
        if memory_size.strip() == "":
            raise ValueError(
                f"slurm.cpu_node.memory.{job_key} must not be an empty string."
            )
        return memory_size

    @property
    def gpu_node_partition(self) -> str:
        """
        Retrieves the partition name for the GPU node from the SLURM configuration.
        This method accesses the SLURM configuration dictionary to fetch the partition
        name for the GPU node. It ensures that the retrieved value is of the correct type.
        Returns:
            str: The partition name for the GPU node.
        Raises:
            TypeError: If the partition value is not of type str.
        """

        partition = self.config["slurm"]["gpu_node"]["partition"]
        self.ensure_config_value_has_correct_type(partition, str)
        return partition

    @property
    def gpu_node_job_header(self) -> str:
        """
        Retrieves the job header configuration for a GPU node.
        This method fetches the job header configuration from the SLURM GPU node
        settings in the configuration file and ensures that the value is of the correct type.
        Returns:
            str: The job header configuration for a GPU node.
        Raises:
            TypeError: If the job header configuration is not of type str.
        """

        job_header = self.config["slurm"]["gpu_node"]["job_header"]
        self.ensure_config_value_has_correct_type(job_header, str)
        return job_header

    def gpu_node_memory_size(self, job_key: str) -> str:
        """
        Retrieve the GPU-node memory request for one named job type.

        Args:
            job_key: Key under slurm.gpu_node.memory that identifies the job
                type requesting memory.

        Returns:
            str: Slurm-compatible memory request, for example 400GB.

        Raises:
            TypeError: If job_key or the configured memory value is not a string,
                or if the memory mapping is not a dictionary.
            KeyError: If the configuration does not define memory for job_key.
            ValueError: If the configured memory value is empty.
        """

        self.ensure_config_value_has_correct_type(job_key, str)
        memory_by_job = self.config["slurm"]["gpu_node"]["memory"]
        self.ensure_config_value_has_correct_type(memory_by_job, dict)
        if job_key not in memory_by_job:
            raise KeyError(
                "Missing GPU-node memory configuration for job type "
                f"'{job_key}' under slurm.gpu_node.memory."
            )
        memory_size = memory_by_job[job_key]
        self.ensure_config_value_has_correct_type(memory_size, str)
        if memory_size.strip() == "":
            raise ValueError(
                f"slurm.gpu_node.memory.{job_key} must not be an empty string."
            )
        return memory_size

    @property
    def gpu_node_gres(self) -> dict:
        """
        Retrieves the GPU node GRES (Generic Resource) configuration from the SLURM configuration.
        This method accesses the SLURM configuration dictionary to fetch the GPU node GRES settings.
        It ensures that the retrieved configuration value is of the correct type (dictionary).
        Returns:
            dict: The GPU node GRES configuration.
        Raises:
            TypeError: If the configuration value is not of the expected type (dictionary).
        """

        gres = self.config["slurm"]["gpu_node"]["gres"]
        # self.ensure_config_value_has_correct_type(gres, dict)
        return gres

    @property
    def slurm_email(self) -> str:
        """
        Retrieve the SLURM email address from the configuration.
        This method fetches the email address specified in the SLURM configuration
        and ensures that it is of the correct type (string).
        Returns:
            str: The email address configured for SLURM notifications.
        Raises:
            KeyError: If the 'slurm' or 'email' key is not found in the configuration.
            TypeError: If the email address is not of type string.
        """

        email = self.config["slurm"]["email"]
        self.ensure_config_value_has_correct_type(email, str)
        return email

    @property
    def slurm_train_time(self) -> str:
        """
        Retrieves the SLURM training time from the configuration.
        This method fetches the training time specified in the SLURM configuration
        and ensures that it is of the correct type (string).
        Returns:
            str: The SLURM training time as a string.
        Raises:
            TypeError: If the SLURM training time is not a string.
        """

        slurm_time = self.config["slurm"]["train_time"]
        self.ensure_config_value_has_correct_type(slurm_time, str)
        return slurm_time

    @property
    def slurm_test_time(self) -> str:
        """
        Retrieves the SLURM test time from the configuration.
        This method fetches the SLURM test time value from the configuration
        dictionary and ensures that it is of the correct type (string).
        Returns:
            str: The SLURM test time value from the configuration.
        Raises:
            TypeError: If the SLURM test time value is not a string.
        """

        slurm_time = self.config["slurm"]["test_time"]
        self.ensure_config_value_has_correct_type(slurm_time, str)
        return slurm_time

    @property
    def slurm_preprocess_time(self) -> str:
        """
        Retrieves the SLURM preprocess time from the configuration.
        This method fetches the SLURM preprocess time value from the configuration
        dictionary and ensures that it is of the correct type (string).
        Returns:
            str: The SLURM preprocess time value from the configuration.
        Raises:
            TypeError: If the SLURM preprocess time value is not a string.
        """

        slurm_time = self.config["slurm"]["preprocess_time"]
        self.ensure_config_value_has_correct_type(slurm_time, str)
        return slurm_time

    def _experiment_config(self) -> dict:
        """Return experiment settings from the portable or legacy YAML block."""
        if "experiment" in self.config:
            return self.config["experiment"]
        return self.config["slurm"]

    @property
    def experiment_name(self) -> str:
        experiment_name = self._experiment_config()["experiment_name"]
        self.ensure_config_value_has_correct_type(experiment_name, str)
        return experiment_name

    @property
    def run_name(self) -> str:
        run_name = self._experiment_config()["run_name"]
        self.ensure_config_value_has_correct_type(run_name, str)
        return run_name

    @property
    def experiments_dir(self) -> str:
        experiments_dir = self._experiment_config()["experiments_dir"]
        self.ensure_config_value_has_correct_type(experiments_dir, str)
        return experiments_dir

    @property
    def slurm_experiment_name(self) -> str:
        """Compatibility accessor for experiment names in existing callers."""
        return self.experiment_name

    @property
    def slurm_run_name(self) -> str:
        """Compatibility accessor for run names in existing callers."""
        return self.run_name

    @property
    def slurm_experiments_dir(self) -> str:
        """Compatibility accessor for the configured experiments directory."""
        return self.experiments_dir

    @property
    def container_path(self) -> Path:
        """
        Retrieves the directory path where container images are stored.

        This method reads the container download directory from the
        configuration file and converts it to a `Path` object. It ensures that
        the stored value is a string before returning the normalized path.

        Returns:
            Path: Directory path where container images are stored.

        Raises:
            TypeError: If the configured download path is not a string.
        """

        download_path = self.config["containers"]["download_path"]
        self.ensure_config_value_has_correct_type(download_path, str)
        return Path(download_path)

    @property
    def hdbet_version(self) -> str:
        """
        Retrieves the configured HDBET container version.

        This method reads the HDBET version from the containers section of the
        configuration file and ensures it is a string.

        Returns:
            str: The HDBET container version identifier.

        Raises:
            TypeError: If the configured HDBET version is not a string.
        """

        hdbet_version = self.config["containers"]["hdbet_version"]
        self.ensure_config_value_has_correct_type(hdbet_version, str)
        return hdbet_version

    @property
    def fsl_version(self) -> str:
        """
        Retrieves the configured FSL container version.

        This method reads the FSL version from the containers section of the
        configuration file and ensures it is a string.

        Returns:
            str: The FSL container version identifier.

        Raises:
            TypeError: If the configured FSL version is not a string.
        """

        fsl_version = self.config["containers"]["fsl_version"]
        self.ensure_config_value_has_correct_type(fsl_version, str)
        return fsl_version

    @property
    def environment_seed(self) -> int:
        """
        Retrieves the seed value for the environment from the configuration.
        This method fetches the seed value specified in the environment section
        of the configuration and ensures that it is of the correct type (int).
        Returns:
            int: The seed value for the environment.
        Raises:
            KeyError: If the 'seed' key is not found in the 'environment' section of the configuration.
            TypeError: If the 'seed' value is not of type int.
        """

        seed = self.config["environment"]["seed"]
        self.ensure_config_value_has_correct_type(seed, int)
        return seed

    @property
    def model_architecture(self) -> str:
        """
        Which model architecture to use.

        This paper repository supports the CSNet architecture only.

        Returns:
            str: ``"csnet"``.
        """

        architecture = self.config["model"]["architecture"]
        self.ensure_config_value_has_correct_type(architecture, str)
        if architecture not in constants.MODEL_TYPES:
            raise ValueError(
                f"Model architecture '{architecture}' is not supported. "
                f"Supported type: {constants.CSNET_MODEL_TYPE}."
            )

        return architecture

    @property
    def classification_only(self) -> bool:
        """
        Whether to classify only or also segment (for CSNet).
        Returns:
            bool: whether to classify only.
        """

        classify_only = self.config["model"]["classification_only"]
        self.ensure_config_value_has_correct_type(classify_only, bool)

        return classify_only

    @property
    def train_kfold(self) -> bool:
        """
        Determines if k-fold cross-validation should be used during training.
        This method retrieves the k-fold cross-validation setting from the
        configuration and ensures it is of the correct type (boolean).
        Returns:
            bool: True if k-fold cross-validation is enabled, False otherwise.
        """

        kfold = self.config["model"]["train"]["kfold"]
        self.ensure_config_value_has_correct_type(kfold, bool)
        return kfold

    @property
    def train_num_folds(self) -> int:
        """
        Retrieves the number of folds for training from the configuration.
        This method accesses the configuration dictionary to get the number of folds
        specified for the training process. It also ensures that the retrieved value
        is of the correct type (integer).
        Returns:
            int: The number of folds for training.
        Raises:
            TypeError: If the retrieved configuration value is not of type int.
        """

        num_folds = self.config["model"]["train"]["num_folds"]
        self.ensure_config_value_has_correct_type(num_folds, int)
        return num_folds

    @property
    def train_epochs(self) -> int:
        """
        Retrieve the number of training epochs from the configuration.
        This method accesses the configuration dictionary to fetch the number of
        epochs specified for training the model. It ensures that the retrieved
        value is of the correct type (int) before returning it.
        Returns:
            int: The number of training epochs.
        Raises:
            TypeError: If the retrieved configuration value is not of type int.
        """

        train_epochs = self.config["model"]["train"]["epochs"]
        self.ensure_config_value_has_correct_type(train_epochs, int)
        return train_epochs

    @property
    def train_batch_size(self) -> int:
        """
        Retrieves the training batch size from the configuration.
        This method accesses the configuration dictionary to fetch the batch size
        used for training the model. It ensures that the retrieved value is of the
        correct type (integer) before returning it.
        Returns:
            int: The batch size for training.
        Raises:
            TypeError: If the batch size is not an integer.
        """

        train_batch_size = self.config["model"]["train"]["batch_size"]
        self.ensure_config_value_has_correct_type(train_batch_size, int)
        return train_batch_size

    @property
    def train_dropout_rate(self) -> float:
        """
        Retrieves the dropout rate for training from the configuration.
        This method accesses the configuration dictionary to fetch the dropout rate
        specified for the training model. It ensures that the retrieved value is of
        the correct type (float) before returning it.
        Returns:
            float: The dropout rate for training.
        Raises:
            TypeError: If the dropout rate is not of type float.
        """

        dropout = self.config["model"]["train"]["dropout_rate"]
        self.ensure_config_value_has_correct_type(dropout, float)
        return dropout

    @property
    def train_optimizer_lr(self) -> float:
        """
        Retrieves the learning rate for the optimizer from the configuration.
        This method accesses the configuration dictionary to fetch the learning rate
        specified for the optimizer under the training settings. It ensures that the
        retrieved value is of the correct type (float) before returning it.
        Returns:
            float: The learning rate for the optimizer.
        Raises:
            TypeError: If the learning rate is not of type float.
        """

        optimizer_learning_rate = self.config["model"]["train"]["optimizer"][
            "learning_rate"
        ]
        self.ensure_config_value_has_correct_type(optimizer_learning_rate, float)
        return optimizer_learning_rate

    @property
    def train_optimizer_weight_decay(self) -> float:
        """
        Retrieves the weight decay value for the optimizer from the configuration.
        This method accesses the configuration dictionary to fetch the weight decay
        value specified for the optimizer under the training settings. It ensures
        that the retrieved value is of the correct type (float) before returning it.
        Returns:
            float: The weight decay value for the optimizer.
        Raises:
            TypeError: If the retrieved weight decay value is not of type float.
        """

        optimizer_weight_decay = self.config["model"]["train"]["optimizer"][
            "weight_decay"
        ]
        self.ensure_config_value_has_correct_type(optimizer_weight_decay, float)
        return optimizer_weight_decay

    @property
    def train_optimizer_scheduler_minimum_lr(self) -> float:
        """
        Retrieves the minimum learning rate for the optimizer scheduler from the configuration.
        This method accesses the configuration dictionary to fetch the minimum learning rate
        specified for the optimizer scheduler under the training settings. It ensures that
        the retrieved value is of the correct type (float) before returning it.
        Returns:
            float: The minimum learning rate for the optimizer scheduler.
        Raises:
            TypeError: If the retrieved minimum learning rate is not of type float.
        """

        train_optimizer_reduction = self.config["model"]["train"]["optimizer"][
            "scheduler"
        ]["minimum_lr"]
        self.ensure_config_value_has_correct_type(train_optimizer_reduction, float)
        return train_optimizer_reduction

    @property
    def train_optimizer_scheduler_reduction_factor(self) -> float:
        """
        Retrieves the reduction factor for the optimizer scheduler from the configuration.
        This method accesses the configuration dictionary to fetch the reduction factor
        specified for the optimizer scheduler under the training settings. It ensures that
        the retrieved value is of the correct type (float) before returning it.
        Returns:
            float: The reduction factor for the optimizer scheduler.
        Raises:
            TypeError: If the retrieved reduction factor is not of type float.
        """

        train_optimizer_reduction = self.config["model"]["train"]["optimizer"][
            "scheduler"
        ]["reduction_factor"]
        self.ensure_config_value_has_correct_type(train_optimizer_reduction, float)
        return train_optimizer_reduction

    @property
    def train_optimizer_scheduler_patience(self) -> int:
        """
        Retrieves the patience value for the optimizer scheduler from the configuration.
        This method accesses the configuration dictionary to fetch the patience value
        specified for the optimizer scheduler under the training settings. It ensures that
        the retrieved value is of the correct type (int) before returning it.
        Returns:
            int: The patience value for the optimizer scheduler.
        Raises:
            TypeError: If the retrieved patience value is not of type int.
        """

        train_optimizer_reduction = self.config["model"]["train"]["optimizer"][
            "scheduler"
        ]["patience"]
        self.ensure_config_value_has_correct_type(train_optimizer_reduction, int)
        return train_optimizer_reduction

    @property
    def train_optimizer_scheduler_threshold(self) -> float:
        """
        Retrieves the threshold value for the optimizer scheduler from the configuration.
        This method accesses the configuration dictionary to fetch the threshold value
        specified for the optimizer scheduler under the training settings. It ensures that
        the retrieved value is of the correct type (float) before returning it.
        Returns:
            float: The threshold value for the optimizer scheduler.
        Raises:
            TypeError: If the retrieved threshold value is not of type float.
        """

        train_optimizer_reduction = self.config["model"]["train"]["optimizer"][
            "scheduler"
        ]["threshold"]
        self.ensure_config_value_has_correct_type(train_optimizer_reduction, float)
        return train_optimizer_reduction

    @property
    def train_early_stopping_delta(self) -> Union[int, float]:
        """
        Retrieves the early stopping delta value from the configuration.
        This method fetches the early stopping delta value from the configuration
        dictionary and ensures that it is of the correct type (either int or float).
        Returns:
            Union[int, float]: The early stopping delta value.
        Raises:
            TypeError: If the early stopping delta value is not of type int or float.
        """

        early_stopping_delta = self.config["model"]["train"]["early_stopping"]["delta"]
        self.ensure_config_value_has_correct_type(early_stopping_delta, (int, float))
        return early_stopping_delta

    @property
    def train_early_stopping_patience(self) -> int:
        """
        Retrieves the early stopping patience value from the configuration.
        This method accesses the configuration dictionary to fetch the early stopping
        patience value used during model training. It ensures that the retrieved value
        is of the correct type (int) before returning it.
        Returns:
            int: The early stopping patience value.
        Raises:
            TypeError: If the early stopping patience value is not an integer.
        """

        early_stopping_patience = self.config["model"]["train"]["early_stopping"][
            "patience"
        ]
        self.ensure_config_value_has_correct_type(early_stopping_patience, int)
        return early_stopping_patience

    @property
    def test_kfold(self) -> bool:
        """
        Retrieves the k-fold testing configuration value.
        This method fetches the k-fold testing configuration value from the
        configuration dictionary and ensures it is of the correct type (bool).
        Returns:
            bool: The k-fold testing configuration value.
        Raises:
            TypeError: If the k-fold configuration value is not of type bool.
        """

        kfold = self.config["model"]["test"]["kfold"]
        self.ensure_config_value_has_correct_type(kfold, bool)
        return kfold

    @property
    def test_batch_size(self) -> int:
        """
        Retrieves the batch size for testing from the configuration.
        This method accesses the configuration dictionary to fetch the batch size
        specified for the testing phase of the model. It ensures that the retrieved
        value is of the correct type (integer) before returning it.
        Returns:
            int: The batch size for testing.
        Raises:
            KeyError: If the configuration keys are not found.
            TypeError: If the batch size is not of type int.
        """

        batch_size = self.config["model"]["test"]["batch_size"]
        self.ensure_config_value_has_correct_type(batch_size, int)
        return batch_size

    @property
    def test_num_folds(self) -> int:
        """
        Retrieves the number of folds for testing from the configuration.
        This method accesses the configuration dictionary to fetch the number of folds
        specified for testing under the "model" -> "test" -> "num_folds" path. It also
        ensures that the retrieved value is of the correct type (int).
        Returns:
            int: The number of folds for testing.
        Raises:
            TypeError: If the retrieved configuration value is not of type int.
        """

        num_folds = self.config["model"]["test"]["num_folds"]
        self.ensure_config_value_has_correct_type(num_folds, int)
        return num_folds

    @property
    def test_results_dir(self) -> str:
        """
        Retrieves the directory path for test results from the configuration.
        This method accesses the configuration dictionary to fetch the path
        where test results should be stored. It ensures that the retrieved
        value is of the correct type (string).
        Returns:
            str: The directory path for test results.
        Raises:
            TypeError: If the retrieved configuration value is not a string.
        """

        test_dir = self.config["model"]["test"]["results_dir"]
        self.ensure_config_value_has_correct_type(test_dir, str)
        return test_dir

    @property
    def test_model_dir(self) -> str:
        """
        Retrieves the directory path of the model to use for testing.
        It ensures that the retrieved value is of the correct type (string).
        Returns:
            str: The directory path for test results.
        Raises:
            TypeError: If the retrieved configuration value is not a string.
        """

        test_model = self.config["model"]["test"]["model_dir"]
        if test_model is not None:
            self.ensure_config_value_has_correct_type(test_model, str)
        return test_model

    @property
    def test_model_type(self) -> str:
        """
        Retrieves the type of the model to use for testing. It should be either "best" or "last".
        It ensures that the retrieved value is of the correct type (string).
        Returns:
            str: The type of the model to use for predictions.
        Raises:
            TypeError: If the retrieved configuration value is not a string.
            AssertionError: If the model type is not "best" or "last".
        """

        test_model_type = self.config["model"]["test"]["model_type"]
        self.ensure_config_value_has_correct_type(test_model_type, str)
        assert test_model_type in [
            "best",
            "last",
        ], f"Model type {test_model_type} is not supported. Supported types are 'best' or 'last'."

        return test_model_type

    @property
    def test_save_predictions(self) -> bool:
        """
        Checks if the configuration is set to save predictions during testing.
        This method retrieves the 'save_predictions' flag from the configuration
        under the 'model' -> 'test' section and ensures it is of the correct type (bool).
        Returns:
            bool: True if predictions should be saved during testing, False otherwise.
        """

        save_predictions = self.config["model"]["test"]["save_predictions"]
        self.ensure_config_value_has_correct_type(save_predictions, bool)
        return save_predictions

    @property
    def test_save_prob_map(self) -> bool:
        """
        Check whether voxelwise segmentation probability maps should be saved.

        This property reads the save_prob_map flag from the model test section
        of the configuration. When enabled, inference writes the softmax
        segmentation probability map for each case in the predictions folder.
        The exported maps are intended for downstream uncertainty workflows,
        such as Deep Ensemble aggregation across independently trained models.

        Returns:
            bool: True when inference should save segmentation probability maps.

        Raises:
            TypeError: If the configured value is not a boolean.
        """

        save_prob_map = self.config["model"]["test"]["save_prob_map"]
        self.ensure_config_value_has_correct_type(save_prob_map, bool)
        return save_prob_map

    @property
    def mc_dropout_samples(self) -> int:
        """
        Retrieves the number of Monte Carlo dropout samples from the configuration.
        This method accesses the configuration dictionary to fetch the number of
        Monte Carlo (MC) dropout samples specified under the "UQ" (Uncertainty Quantification)
        section. It ensures that the retrieved value is of the correct type (int) before returning it.
        Returns:
            int: The number of MC dropout samples as specified in the configuration.
        Raises:
            TypeError: If the retrieved configuration value is not of type int.
        """

        mc_samples = self.config["model"]["UQ"]["mcd"]["n_samples"]
        self.ensure_config_value_has_correct_type(mc_samples, int)
        return mc_samples

    @property
    def mc_dropout_rate(self) -> float:
        """
        Retrieves the dropout rate for Monte Carlo dropout from the configuration.
        This method accesses the configuration dictionary to fetch the dropout rate
        specified for Monte Carlo (MC) dropout under the "UQ" (Uncertainty Quantification)
        section. It ensures that the retrieved value is of the correct type (float) before returning it.
        Returns:
            float: The dropout rate for MC dropout as specified in the configuration.
        Raises:
            TypeError: If the retrieved configuration value is not of type float.
        """

        mc_dropout_rate = self.config["model"]["UQ"]["mcd"]["dropout_rate"]
        self.ensure_config_value_has_correct_type(mc_dropout_rate, float)
        return mc_dropout_rate

    @property
    def output_format_config(self) -> dict:
        """
        Retrieve the generated image and table output-format configuration.

        This top-level block defines the filename extensions used for generated
        figures and tabular outputs throughout inference and uncertainty
        quantification. Input files, such as labels tables, are unaffected.

        Returns:
            dict: Output-format configuration containing images and tables.

        Raises:
            TypeError: If the configured output-format block is not a dictionary.
        """
        output_format = self.config["output_format"]
        self.ensure_config_value_has_correct_type(output_format, dict)
        return output_format

    @property
    def output_images_format(self) -> str:
        """Retrieve the filename extension for generated figures.

        Returns:
            str: Image extension, including its leading period.

        Raises:
            TypeError: If the configured image extension is not a string.
            ValueError: If the configured image extension is unsupported.
        """
        image_format = self.output_format_config["images"]
        self.ensure_config_value_has_correct_type(image_format, str)
        if image_format not in constants.SUPPORTED_IMAGE_OUTPUT_FORMATS:
            supported_formats = ", ".join(constants.SUPPORTED_IMAGE_OUTPUT_FORMATS)
            raise ValueError(
                "Unsupported output_format.images value "
                f"'{image_format}'. Choose one of: {supported_formats}."
            )
        return image_format

    @property
    def output_tables_format(self) -> str:
        """Retrieve the filename extension for generated tables.

        Returns:
            str: Table extension, including its leading period.

        Raises:
            TypeError: If the configured table extension is not a string.
            ValueError: If the configured table extension is unsupported.
        """
        table_format = self.output_format_config["tables"]
        self.ensure_config_value_has_correct_type(table_format, str)
        if table_format not in constants.SUPPORTED_TABLE_OUTPUT_FORMATS:
            supported_formats = ", ".join(constants.SUPPORTED_TABLE_OUTPUT_FORMATS)
            raise ValueError(
                "Unsupported output_format.tables value "
                f"'{table_format}'. Choose one of: {supported_formats}."
            )
        return table_format

    def uq_method_config(self, method: str) -> dict:
        """
        Retrieve the configuration block for one uncertainty quantification method.

        The project now supports multiple UQ methods, each with its own nested
        configuration block under model -> UQ. This helper provides a single
        access point for those method-specific settings so the rest of the code
        does not need to hardcode any particular method name.

        Args:
            method: Method key such as mcd, de, or mcd_de.

        Returns:
            dict: Configuration block for the requested UQ method.

        Raises:
            KeyError: If the requested UQ method is not present in the config.
            TypeError: If the stored method configuration is not a dictionary.
        """
        uq_config = self.config["model"]["UQ"]
        if method not in uq_config:
            raise KeyError(f"Missing UQ configuration block for '{method}'")
        method_config = uq_config[method]
        self.ensure_config_value_has_correct_type(method_config, dict)
        return method_config

    def uq_method_num_models(self, method: str) -> int:
        """
        Retrieve the number of ensemble models configured for one UQ method.

        This is used by deterministic ensemble runs and mixed methods that
        combine ensembles with dropout. Methods that do not define num_models
        will raise an explicit error so missing configuration is caught early.

        Args:
            method: Method key such as de or mcd_de.

        Returns:
            int: Number of ensemble members for the requested method.

        Raises:
            KeyError: If the method block does not define num_models.
            TypeError: If the configured value is not an integer.
        """
        method_config = self.uq_method_config(method)
        if "num_models" not in method_config:
            raise KeyError(f"Missing 'num_models' for UQ method '{method}'")
        num_models = method_config["num_models"]
        self.ensure_config_value_has_correct_type(num_models, int)
        return num_models

    def uq_method_num_samples(self, method: str) -> int:
        """
        Retrieve the number of Monte Carlo samples configured for one UQ method.

        This is the method-agnostic access point for sample-count parameters.
        For Monte Carlo dropout it maps to n_samples; for mixed methods it
        can be used alongside the ensemble size to retrieve the full setting.

        Args:
            method: Method key such as mcd or mcd_de.

        Returns:
            int: Number of stochastic forward passes for the method.

        Raises:
            KeyError: If the method block does not define n_samples.
            TypeError: If the configured value is not an integer.
        """
        method_config = self.uq_method_config(method)
        if "n_samples" not in method_config:
            raise KeyError(f"Missing 'n_samples' for UQ method '{method}'")
        num_samples = method_config["n_samples"]
        self.ensure_config_value_has_correct_type(num_samples, int)
        return num_samples

    def uq_method_dropout_rate(self, method: str) -> float:
        """
        Retrieve the dropout rate configured for one UQ method.

        The dropout rate is shared by methods that rely on stochastic dropout,
        and it remains explicit in the configuration so that all runs are
        reproducible and tied to the exact experiment settings.

        Args:
            method: Method key such as mcd or mcd_de.

        Returns:
            float: Dropout rate for the requested method.

        Raises:
            KeyError: If the method block does not define dropout_rate.
            TypeError: If the configured value is not numeric.
        """
        method_config = self.uq_method_config(method)
        if "dropout_rate" not in method_config:
            raise KeyError(f"Missing 'dropout_rate' for UQ method '{method}'")
        dropout_rate = method_config["dropout_rate"]
        self.ensure_config_value_has_correct_type(dropout_rate, float)
        return dropout_rate

    def uq_method_ensemble_root_dir(self, method: str) -> str:
        """
        Retrieve the root directory containing ensemble member runs.

        Deep ensemble methods need to locate several independently trained
        model runs. This directory is the common parent that contains one
        subdirectory per ensemble member. The exact subdirectory names are
        defined by the seed folder template for the same method.

        Args:
            method: Method key such as de or mcd_de.

        Returns:
            str: Directory containing all configured ensemble member folders.

        Raises:
            KeyError: If the method block does not define ensemble_root_dir.
            TypeError: If the configured value is not a string.
        """
        method_config = self.uq_method_config(method)
        if "ensemble_root_dir" not in method_config:
            raise KeyError(f"Missing 'ensemble_root_dir' for UQ method '{method}'")
        ensemble_root_dir = method_config["ensemble_root_dir"]
        self.ensure_config_value_has_correct_type(ensemble_root_dir, str)
        return ensemble_root_dir

    def uq_method_seed_folder_template(self, method: str) -> str:
        """
        Retrieve the folder-name template used for ensemble member runs.

        The template must contain the field {seed}. Each configured seed is
        inserted into this template to locate the corresponding model run
        under ensemble_root_dir.

        Args:
            method: Method key such as de or mcd_de.

        Returns:
            str: Folder template for ensemble members.

        Raises:
            KeyError: If the method block does not define seed_folder_template.
            ValueError: If the template does not contain the {seed} field.
            TypeError: If the configured value is not a string.
        """
        method_config = self.uq_method_config(method)
        if "seed_folder_template" not in method_config:
            raise KeyError(f"Missing 'seed_folder_template' for UQ method '{method}'")
        seed_folder_template = method_config["seed_folder_template"]
        self.ensure_config_value_has_correct_type(seed_folder_template, str)
        if "{seed}" not in seed_folder_template:
            raise ValueError(
                f"seed_folder_template for UQ method '{method}' must contain "
                "the {seed} field."
            )
        return seed_folder_template

    def uq_method_seeds(self, method: str) -> list[int]:
        """
        Retrieve the explicit ensemble seed list for one UQ method.

        The order of this list is preserved when the ensemble is aggregated.
        This makes the selected ensemble members reproducible and avoids
        accidentally including extra run folders that happen to exist on disk.

        Args:
            method: Method key such as de or mcd_de.

        Returns:
            list[int]: Seed values identifying ensemble member folders.

        Raises:
            KeyError: If the method block does not define seeds.
            ValueError: If the list is empty.
            TypeError: If seeds is not a list or contains non-integer values.
        """
        method_config = self.uq_method_config(method)
        if "seeds" not in method_config:
            raise KeyError(f"Missing 'seeds' for UQ method '{method}'")
        seeds = method_config["seeds"]
        self.ensure_config_value_has_correct_type(seeds, list)
        if not seeds:
            raise ValueError(f"seeds for UQ method '{method}' cannot be empty.")
        for seed in seeds:
            self.ensure_config_value_has_correct_type(seed, int)
        return seeds

    @property
    def mask_based_seg_unc_config(self) -> dict:
        """
        Retrieve the configuration block for mask-based segmentation UQ.

        This section controls how segmentation uncertainty maps are aggregated
        into case-level scores, including boundary weighting, dilation, and
        export behavior.

        Returns:
            dict: Mask-based segmentation uncertainty configuration block.
        """
        mask_based_config = self.config["model"]["UQ"]["mask_based_segmentation"]
        self.ensure_config_value_has_correct_type(mask_based_config, dict)
        return mask_based_config

    @property
    def mask_based_seg_unc_dilation_radius_voxels(self) -> int:
        """
        Retrieve the dilation radius used for the predicted-tumor mask.

        The radius is expressed in voxels and is used to expand the predicted
        tumor region when computing tumor-focused uncertainty summaries.

        Returns:
            int: Dilation radius in voxels.
        """
        radius = self.mask_based_seg_unc_config["dilation_radius_voxels"]
        self.ensure_config_value_has_correct_type(radius, int)
        return radius

    @property
    def mask_based_seg_unc_boundary_sigma_voxels(self) -> float:
        """
        Retrieve the Gaussian sigma used for boundary-weighted aggregation.

        The sigma is expressed in voxels and determines how quickly the
        boundary weighting decays away from the tumor contour.

        Returns:
            float: Boundary-weighting sigma in voxels.
        """
        sigma = self.mask_based_seg_unc_config["boundary_sigma_voxels"]
        self.ensure_config_value_has_correct_type(sigma, (int, float))
        return float(sigma)

    @property
    def mask_based_seg_unc_export_maps(self) -> bool:
        """
        Determine whether mask-based UQ should export uncertainty maps.

        When enabled, the aggregation code writes NIfTI maps to disk in
        addition to the case-level table summaries.

        Returns:
            bool: True when uncertainty maps should be exported.
        """
        export_maps = self.mask_based_seg_unc_config["export_maps"]
        self.ensure_config_value_has_correct_type(export_maps, bool)
        return export_maps

    @property
    def mask_based_seg_unc_export_masks(self) -> bool:
        """
        Determine whether mask-based UQ should export the masks used for scoring.

        This controls whether ground-truth and predicted segmentation masks are
        written out alongside the uncertainty maps and table outputs.

        Returns:
            bool: True when masks should be exported.
        """
        export_masks = self.mask_based_seg_unc_config["export_masks"]
        self.ensure_config_value_has_correct_type(export_masks, bool)
        return export_masks

    @property
    def mask_based_seg_unc_ground_truth_mask_relative_path(self) -> str:
        """
        Retrieve the relative path to the ground-truth mask file.

        This path is interpreted relative to each case directory and is used
        to locate the reference segmentation for mask-based uncertainty
        aggregation.

        Returns:
            str: Relative path to the ground-truth mask.
        """
        ground_truth_mask_relative_path = self.mask_based_seg_unc_config[
            "ground_truth_mask_relative_path"
        ]
        self.ensure_config_value_has_correct_type(ground_truth_mask_relative_path, str)
        return ground_truth_mask_relative_path

    @property
    def mask_based_seg_unc_brain_mask_relative_path(self) -> str:
        """
        Retrieve the relative path to the brain mask file.

        The brain mask is used to restrict the region of interest when scoring
        uncertainty over the full brain volume.

        Returns:
            str: Relative path to the brain mask.
        """
        brain_mask_relative_path = self.mask_based_seg_unc_config[
            "brain_mask_relative_path"
        ]
        self.ensure_config_value_has_correct_type(brain_mask_relative_path, str)
        return brain_mask_relative_path

    def has_structural(self) -> bool:
        """
        Return whether at least one structural MRI modality is enabled.
        """
        return any(
            modality in self.data_modalities for modality in constants.SCAN_TYPES
        )

    def has_modality(self, modality: str) -> bool:
        """
        Return whether the specified modality is enabled.
        """
        self.ensure_config_value_has_correct_type(modality, str)
        return modality in self.data_modalities
