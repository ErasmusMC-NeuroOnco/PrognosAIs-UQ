import datetime
import shutil
import traceback
from pathlib import Path

import itk
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed

import prognosais.IO.constants as constants
import prognosais.preprocessing.utils as utils


class Preprocessor:
    """
    Preprocess patient folders into registered and cropped NIfTI outputs.

    The preprocessor loads scans and masks, optionally performs registration
    and bias-field correction, and writes the processed outputs back into the
    patient directory structure.
    """

    def __init__(
        self,
        data_dir: Path,
        this_script_dir: Path,
        mask_origin_path: Path | None,
        already_registered: bool = False,
        bias_field_correct: bool = True,
    ) -> None:
        """
        Initialize the Preprocessor class with paths and settings.

        Args:
            data_dir: Path to the patient data directory.
            this_script_dir: Path to the preprocessing code directory.
            mask_origin_path: Path to the mask origin file, if available.
            already_registered: If True, load registered scans instead of raw
                NIfTI files.
            bias_field_correct: If True, apply bias-field correction.
        """
        self.data_dir = data_dir
        self.failed_dir = data_dir.parent.joinpath(constants.FAILED_PATIENTS_FOLDER)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        self.atlas_paths = {
            "T1": this_script_dir.joinpath(
                constants.MNI_ATLAS_FOLDER, constants.T1_MNI_ATLAS
            ),
            "T2": this_script_dir.joinpath(
                constants.MNI_ATLAS_FOLDER, constants.T2_MNI_ATLAS
            ),
        }
        self.brain_mask = sitk.ReadImage(
            str(
                this_script_dir.joinpath(
                    constants.MNI_ATLAS_FOLDER, constants.MNI_ATLAS_MASK
                )
            ),
            sitk.sitkUInt8,
        )
        self.rigid_parameter_file_path = this_script_dir.joinpath(
            constants.PARAMETER_FOLDER, constants.ELASTIX_RIGID_PARAMETER_MAP_FILE
        )
        self.affine_parameter_file_path = this_script_dir.joinpath(
            constants.PARAMETER_FOLDER, constants.ELASTIX_AFFINE_PARAMETER_MAP_FILE
        )
        self.already_registered = already_registered
        self.bias_field_correct = bias_field_correct

        # Placeholder for intermediate results
        self.current_patient_scans = {}
        self.current_patient_tumor_mask = None
        self.has_tumor_mask = True
        self.mask_dict = {}

        # The origin modality is needed only to transform a raw tumor mask.
        # Already-registered masks are read directly from REGISTERED/.
        if mask_origin_path and not already_registered:
            df = pd.read_csv(mask_origin_path, sep="\t")
            self.mask_dict = df.set_index("patient").to_dict()["scan"]

    def process_all_patients(self) -> None:
        """
        Process every patient directory found under the input root.
        """
        patients = [
            i_patient
            for i_patient in sorted(self.data_dir.iterdir())
            if i_patient.is_dir()
        ]

        n_pats = len(patients)
        if n_pats == 0:
            raise ValueError(f"No patient directories found in {self.data_dir}")
        n_cpu = utils.get_number_of_threads()
        n_jobs = min(constants.N_JOBS_PREPROCESSING, n_pats, n_cpu)

        print(
            f"Starting {n_jobs} parallel processes for {n_pats} patients, on {n_cpu} threads"
        )

        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(self.process_patient)(i_patient) for i_patient in patients
        )
        n_failed = results.count(False)
        if n_failed:
            raise RuntimeError(
                f"Preprocessing failed for {n_failed} of {n_pats} patients. "
                f"Inspect {self.failed_dir}."
            )
        print("All patients processed successfully")

    def process_patient(self, patient_dir: Path) -> bool:
        """
        Process a single patient's data directory.

        Args:
            patient_dir: Directory containing one patient's data.

        Returns:
            True if every step completed; False after moving a failed case.
        """
        cur_dir = patient_dir.name
        self.has_tumor_mask = True
        # Load tasks
        if not self.already_registered:
            jobdict = {
                "Scans/mask loading": lambda: (
                    setattr(
                        self,
                        "current_patient_scans",
                        self.load_images_and_mask(patient_dir, constants.NIFTI_FOLDER)[
                            0
                        ],
                    ),
                    setattr(
                        self,
                        "current_patient_tumor_mask",
                        self.load_images_and_mask(patient_dir, constants.NIFTI_FOLDER)[
                            1
                        ],
                    ),
                ),
            }
            jobdict["Scan registration"] = lambda: self.register_scans(patient_dir)
        else:
            jobdict = {
                "Scans/mask loading": lambda: (
                    setattr(
                        self,
                        "current_patient_scans",
                        self.load_images_and_mask(
                            patient_dir, constants.REGISTERED_FOLDER, load_as_sitk=True
                        )[0],
                    ),
                    setattr(
                        self,
                        "current_patient_tumor_mask",
                        self.load_images_and_mask(
                            patient_dir, constants.REGISTERED_FOLDER, load_as_sitk=True
                        )[1],
                    ),
                ),
            }

        if self.bias_field_correct:
            jobdict["Biasfield correction"] = lambda: self.apply_biasfield_correction(
                patient_dir
            )

        jobdict["Preprocessing"] = lambda: self.apply_preprocessing(patient_dir)

        # Execute tasks
        for task_name, task_func in jobdict.items():
            print(f"{task_name} started for patient {cur_dir}", flush=True)
            try:
                task_func()
            except Exception as e:
                print(f"{task_name} failed for patient {cur_dir}: {e}", flush=True)
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                failure_report_file = patient_dir.joinpath(
                    constants.FAILURE_REPORT_FILE.format(exact_time=now)
                )
                with open(failure_report_file, "w") as f:
                    f.write(f"Patient: {cur_dir}\n")
                    f.write(f"Task failed: {task_name}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write(f"Timestamp: {now}\n\n")
                    f.write("Full traceback:\n")
                    f.write(traceback.format_exc())

                print(f"Moving {cur_dir} to 'failed_patients' folder", flush=True)
                shutil.move(str(patient_dir), str(self.failed_dir / cur_dir))
                return False
            print(f"Finished processing {cur_dir}", flush=True)
        return True

    def load_images_and_mask(
        self, patient_dir: Path, dir_name: str, load_as_sitk: bool = False
    ) -> tuple[dict[str, object], object | None]:
        """
        Load MRI scan images and tumor mask for a given patient directory.

        Args:
            patient_dir: Directory containing the patient's data.
            dir_name: Subdirectory within the patient folder to load from.
            load_as_sitk: If True, load images as SimpleITK objects.

        Returns:
            tuple[dict[str, object], object | None]: Loaded scans and tumor
            mask, if present.
        """
        source_dir = patient_dir.joinpath(dir_name)
        images = {}
        for i_modality in constants.SCAN_TYPES:
            image_path = source_dir.joinpath(i_modality).with_suffix(
                constants.DATA_NIFTI_EXTENSION
            )
            if load_as_sitk:
                image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
            else:
                image = itk.imread(str(image_path), itk.F)
            images[i_modality] = image

        tumor_mask_path = source_dir.joinpath(constants.MASK_FILE_NAME).with_suffix(
            constants.DATA_NIFTI_EXTENSION
        )
        if tumor_mask_path.exists():
            self.has_tumor_mask = True
            if load_as_sitk:
                tumor_mask = sitk.ReadImage(str(tumor_mask_path), sitk.sitkUInt8)
            else:
                tumor_mask = itk.imread(str(tumor_mask_path), itk.US)
        else:
            self.has_tumor_mask = False
            tumor_mask = None
            print(
                f"WARNING: Tumor mask is missing for patient {patient_dir.name}",
                flush=True,
            )

        return images, tumor_mask

    def register_scans(self, patient_dir: Path) -> None:
        """
        Register NIFTI files to MNI152 atlas space, save to REGISTERED folder.

        Args:
            patient_dir: Directory containing the patient's data.
        """
        output_dir = patient_dir.joinpath(constants.REGISTERED_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_output_dir = patient_dir.joinpath(constants.ELASTIX_FOLDER)
        txt_output_dir.mkdir(parents=True, exist_ok=True)

        mask_parameter_map = None
        mask_scan = self.mask_dict.get(patient_dir.name, None)
        if self.has_tumor_mask and mask_scan not in (*constants.SCAN_TYPES, "ALL"):
            raise ValueError(
                f"Tumor mask for {patient_dir.name} needs a valid origin modality "
                "(T1, T1CE, T2, FLAIR, or ALL) in mask_origin_file_path "
                "before registration."
            )

        registered_itk_images = {}

        for i_modality, scan in self.current_patient_scans.items():
            atlas_path = (
                self.atlas_paths[constants.T1_MODALITY]
                if i_modality.startswith(constants.T1_MODALITY)
                else self.atlas_paths[constants.T2_MODALITY]
            )
            atlas = itk.imread(str(atlas_path), itk.F)

            # Setup Elastix
            elastix = itk.ElastixRegistrationMethod[
                itk.Image[itk.F, 3], itk.Image[itk.F, 3]
            ].New()
            elastix.SetFixedImage(atlas)
            elastix.SetMovingImage(scan)

            param_obj = itk.ParameterObject.New()
            param_obj.AddParameterFile(str(self.rigid_parameter_file_path))
            param_obj.AddParameterFile(str(self.affine_parameter_file_path))
            elastix.SetParameterObject(param_obj)

            # Logging
            elastix.SetLogToConsole(False)
            elastix.SetLogToFile(True)
            elastix.SetOutputDirectory(str(txt_output_dir))
            elastix.SetLogFileName(
                constants.ELASTIX_LOGFILE.format(modality=i_modality)
            )

            # Perform registration
            elastix.Update()
            result_image = elastix.GetOutput()
            result_params = elastix.GetTransformParameterObject()

            # Save image and parameter file
            itk.imwrite(
                result_image,
                str(
                    output_dir.joinpath(i_modality).with_suffix(
                        constants.DATA_NIFTI_EXTENSION
                    )
                ),
            )
            for i in range(result_params.GetNumberOfParameterMaps()):
                pm = result_params.GetParameterMap(i)
                out_file = txt_output_dir.joinpath(
                    constants.ELASTIX_PARAMETERS.format(0, modality=i_modality)
                )
                itk.ParameterObject.WriteParameterFile(pm, str(out_file))

            registered_itk_images[i_modality] = result_image

            # Store parameter map for tumor mask registration
            if mask_scan == i_modality or (
                mask_scan == "ALL" and i_modality == constants.FLAIR_MODALITY
            ):
                # Set parameter map for tumor mask transformation equal to the map of origin scan
                # In case origin is "ALL", use parameter map of FLAIR
                mask_parameter_map = result_params

        if mask_parameter_map and self.has_tumor_mask:
            for i in range(mask_parameter_map.GetNumberOfParameterMaps()):
                pm = mask_parameter_map.GetParameterMap(i)
                pm["FinalBSplineInterpolationOrder"] = ["0"]
                mask_parameter_map.SetParameterMap(i, pm)

            # Register tumor mask using transformix
            transformix = itk.TransformixFilter[
                type(self.current_patient_tumor_mask)
            ].New()
            transformix.SetTransformParameterObject(mask_parameter_map)
            transformix.SetMovingImage(self.current_patient_tumor_mask)

            transformix.SetLogToConsole(False)
            transformix.SetOutputDirectory(str(txt_output_dir))

            transformix.Update()
            transformed_tumor_mask = transformix.GetOutput()

            itk.imwrite(
                transformed_tumor_mask,
                str(
                    output_dir.joinpath(constants.MASK_FILE_NAME).with_suffix(
                        constants.DATA_NIFTI_EXTENSION
                    )
                ),
            )
            registered_itk_tumor_mask = transformed_tumor_mask
        else:
            registered_itk_tumor_mask = self.current_patient_tumor_mask

        # Convert ITK → SITK here for downstream steps
        sitk_images, sitk_tumor_mask = self.itk_to_sitk_dict(
            registered_itk_images, registered_itk_tumor_mask
        )
        self.current_patient_scans = sitk_images
        self.current_patient_tumor_mask = sitk_tumor_mask

    def itk_to_sitk_dict(
        self, itk_images: dict[str, object], itk_tumor_mask: object | None
    ) -> tuple[dict[str, sitk.Image], sitk.Image | None]:
        """
        Convert a dict of ITK images and a mask into SimpleITK images.

        Args:
            itk_images: Dictionary mapping modality name to ITK image.
            itk_tumor_mask: ITK tumor mask.

        Returns:
            tuple[dict[str, sitk.Image], sitk.Image | None]: Converted scans
            and tumor mask.
        """
        sitk_images = {}
        for i_modality, itk_img in itk_images.items():
            arr = itk.GetArrayFromImage(itk_img)
            sitk_img = sitk.GetImageFromArray(arr)
            sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
            sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
            direction = list(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
            sitk_img.SetDirection(direction)
            sitk_images[i_modality] = sitk.Cast(sitk_img, sitk.sitkFloat32)

        # Mask
        if self.has_tumor_mask:
            mask_arr = itk.GetArrayFromImage(itk_tumor_mask)
            sitk_tumor_mask = sitk.GetImageFromArray(mask_arr)
            sitk_tumor_mask.SetOrigin(tuple(itk_tumor_mask.GetOrigin()))
            sitk_tumor_mask.SetSpacing(tuple(itk_tumor_mask.GetSpacing()))
            sitk_tumor_mask.SetDirection(
                list(itk.GetArrayFromMatrix(itk_tumor_mask.GetDirection()).flatten())
            )
            sitk_tumor_mask = sitk.Cast(sitk_tumor_mask, sitk.sitkUInt32)
        else:
            sitk_tumor_mask = None

        return sitk_images, sitk_tumor_mask

    def apply_biasfield_correction(self, patient_dir: Path) -> None:
        """
        Apply N4 bias field correction to scans and save to BIASFIELD_CORRECTED folder.

        Args:
            patient_dir: Directory containing the patient's data.
        """
        output_dir = patient_dir.joinpath(constants.BIASFIELD_CORRECTED_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)

        t1_scan = self.current_patient_scans.get(constants.T1_MODALITY)
        t1_scan_shrink = sitk.Shrink(t1_scan, [2] * t1_scan.GetDimension())

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        _ = corrector.Execute(t1_scan_shrink)

        for i_modality, scan in self.current_patient_scans.items():
            log_bias_field = corrector.GetLogBiasFieldAsImage(scan)
            corrected_scan = scan / sitk.Exp(log_bias_field)

            sitk.WriteImage(
                corrected_scan,
                str(
                    output_dir.joinpath(i_modality).with_suffix(
                        constants.DATA_NIFTI_EXTENSION
                    )
                ),
            )
            self.current_patient_scans[i_modality] = corrected_scan

    def apply_preprocessing(self, patient_dir: Path) -> None:
        """
        Apply further preprocessing functions and save to PREPROCESSED folder.

        Args:
            patient_dir: Directory containing the patient's data.
        """
        output_dir = patient_dir.joinpath(constants.PREPROCESSED_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i_modality, scan in self.current_patient_scans.items():
            # Skull stripping
            masked_scan = sitk.Mask(scan, self.brain_mask)
            # Cropping to bounding box
            cropped_scan, cropped_brain_mask = utils.crop_to_mask(
                masked_scan, self.brain_mask
            )
            # Normalizing
            normalized_scan = utils.normalize_intensity_with_mask(
                cropped_scan, cropped_brain_mask
            )
            # Minimizing background
            minimized_bg_scan = utils.mask_background_to_min(
                normalized_scan, cropped_brain_mask
            )
            # Save preprocessed scan
            sitk.WriteImage(
                minimized_bg_scan,
                str(
                    output_dir.joinpath(i_modality).with_suffix(
                        constants.DATA_NIFTI_EXTENSION
                    )
                ),
            )

        if self.has_tumor_mask:
            # Mask tumor mask with brain mask first
            masked_tumor_mask = sitk.Mask(
                self.current_patient_tumor_mask, self.brain_mask
            )
            # Collapsing labels of tumor mask
            collapsed_tumor_mask = utils.collapse_mask_labels(masked_tumor_mask)
            # Cropping tumor mask to bounding box
            cropped_tumor_mask, _ = utils.crop_to_mask(
                collapsed_tumor_mask, self.brain_mask
            )

            # Save preprocessed tumor mask
            sitk.WriteImage(
                cropped_tumor_mask,
                str(
                    output_dir.joinpath(constants.MASK_FILE_NAME).with_suffix(
                        constants.DATA_NIFTI_EXTENSION
                    )
                ),
            )
