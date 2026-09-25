from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import SimpleITK as sitk
from spython.main import Client as singularity_client

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
import prognosais.preprocessing.utils as utils


def make_brain_masks(
    input_dir: str | Path,
    data_type: str,
    singularity_container_hdbet_path: str | Path,
    singularity_container_fsl_bet_path: str | Path,
) -> None:
    """
    Create brain masks for each patient in the input directory.
    This function processes each patient's folder, checks the compliance of the scans with HD-BET,
    and applies the appropriate brain mask generation method (HD-BET or FSL BET).
    Args:
        input_dir (str | Path): The directory containing the patient folders.
        data_type (str): The type of data to process. Supported values are
            'original', 'registered', and 'preprocessed'.
        singularity_container_hdbet_path (str | Path): Path to the Singularity container for HD-BET.
        singularity_container_fsl_bet_path (str | Path): Path to the Singularity container for FSL BET.

    Raises:
        ValueError: If data_type is not one of the supported values.
    """

    for i_patient_folder in Path(input_dir).iterdir():

        # Check if it is a directory
        if i_patient_folder.is_dir():

            print(f"Now creating brain masks for patient {i_patient_folder.name}")

            if data_type not in ["original", "registered", "preprocessed"]:
                raise ValueError(
                    "Please specify the data you want to load: "
                    "'original', 'registered', or 'preprocessed'."
                )

            if data_type == "original":
                input_folder = i_patient_folder.joinpath("NIFTI")
                output_folder = i_patient_folder.joinpath("BRAIN_MASKS", "NIFTI")
            elif data_type == "registered":
                input_folder = i_patient_folder.joinpath("REGISTERED")
                output_folder = i_patient_folder.joinpath("BRAIN_MASKS", "REGISTERED")
            elif data_type == "preprocessed":
                input_folder = i_patient_folder.joinpath("FINAL")
                output_folder = i_patient_folder.joinpath("BRAIN_MASKS", "FINAL")
            output_folder.mkdir(parents=True, exist_ok=True)

            # Check compliance for each modality
            hdbet_compliant = {}
            for i_modality in constants.SCAN_TYPES:

                i_input_file = input_folder.joinpath(i_modality).with_suffix(
                    constants.DATA_NIFTI_EXTENSION
                )

                i_scan = sitk.ReadImage(i_input_file)

                spacing = i_scan.GetSpacing()
                size = i_scan.GetSize()
                physical_lengths = [
                    voxel_spacing * voxel_count
                    for voxel_spacing, voxel_count in zip(spacing, size)
                ]

                # Check if scan size is HD-BET compliant
                hdbet_compliant[i_modality] = all(
                    length >= min_length
                    for length, min_length in zip(
                        physical_lengths, constants.HDBET_MIN_LENGTH
                    )
                )

            # Split the modalities into compliant and non-compliant lists
            compliant_modalities = [
                mod for mod, compliant in hdbet_compliant.items() if compliant
            ]
            non_compliant_modalities = [
                mod for mod, compliant in hdbet_compliant.items() if not compliant
            ]

            # Process HD-BET compliant modalities
            if compliant_modalities:
                process_hd_bet_compliant(
                    compliant_modalities,
                    input_folder,
                    output_folder,
                    singularity_container_hdbet_path,
                )

            # Process non-compliant modalities with FSL BET
            if non_compliant_modalities:
                process_non_compliant(
                    non_compliant_modalities,
                    input_folder,
                    output_folder,
                    singularity_container_fsl_bet_path,
                )

            # Print the summary of the processing
            print(f"Patient: {i_patient_folder}")
            print(f"HD-BET processed modalities: {compliant_modalities}")
            print(f"FSL BET processed modalities: {non_compliant_modalities}")


def get_path_as_string(path: Path) -> str:
    """Converts a `Path` object to its absolute string representation.

    Args:
        path (Path): The path object to convert.

    Returns:
        str: The absolute path as a POSIX-compliant string.
    """
    return path.absolute().as_posix()


def process_hd_bet_compliant(
    compliant_modalities: list,
    input_folder: Path,
    output_folder: Path,
    singularity_container: str | Path,
) -> None:
    """
    Process HD-BET compliant modalities by running HD-BET.

    Args:
        compliant_modalities (list): List of compliant modalities to be processed.
        input_folder (Path): Original input folder being processed.
        output_folder (Path): Output folder where the brain masks will be saved.
        singularity_container (str | Path): Path to the Singularity container for HD-BET.
    """

    with tempfile.TemporaryDirectory() as temp_in_folder, tempfile.TemporaryDirectory() as temp_out_folder:
        # Copy HD-BET compliant modalities to temporary input folder
        for i_modality in compliant_modalities:
            shutil.copy2(
                input_folder.joinpath(i_modality).with_suffix(
                    constants.DATA_NIFTI_EXTENSION
                ),
                Path(temp_in_folder)
                .joinpath(i_modality)
                .with_suffix(constants.DATA_NIFTI_EXTENSION),
            )

        print(f"temp input folder: {temp_in_folder}")
        print(f"temp output folder: {temp_out_folder}")

        # Run HD-BET on compliant modalities
        command = ["-i", temp_in_folder, "-o", temp_out_folder]

        print(f"Running HD-BET with command: {command}")
        container_output = singularity_client.run(
            (
                get_path_as_string(singularity_container)
                if isinstance(singularity_container, Path)
                else singularity_container
            ),
            command,
            stream=True,
            nv=True,
            options=["--no-mount", "hostfs"],
        )

        for line in container_output:
            print(line)

        # Copy the results from temporary folder to the appropriate output folder

        for i_modality in compliant_modalities:
            brain_mask_file = i_modality + "_mask" + constants.DATA_NIFTI_EXTENSION
            brain_mask_out_file = (
                constants.BRAIN_MASK_NAME
                + "{modality}"
                + constants.DATA_NIFTI_EXTENSION
            ).format(modality=i_modality)

            shutil.copy2(
                Path(temp_out_folder).joinpath(brain_mask_file),
                output_folder.joinpath(brain_mask_out_file),
            )


def process_non_compliant(
    non_compliant_modalities: list,
    input_folder: Path,
    output_folder: Path,
    fsl_singularity_container: str | Path,
) -> None:
    """
    Create a brain mask using FSL BET for non-HD-BET compliant scans.

    Args:
        non_compliant_modalities (list): List of non-compliant modalities to be processed.
        input_folder (Path): Original input folder being processed.
        output_folder (Path): Output folder where the brain masks will be saved.
        config (configIO.Config): Loaded configuration.
        i_patient_folder (str): Patient folder being processed.
        singularity_container (str | Path): Path to the Singularity container for FSL.
    """

    for i_modality in non_compliant_modalities:

        i_input_file = input_folder.joinpath(i_modality).with_suffix(
            constants.DATA_NIFTI_EXTENSION
        )
        scan_folder = i_input_file.parent
        scan_file_name = i_input_file.name
        scan_file_base = scan_file_name.split(constants.DATA_NIFTI_EXTENSION)[0]

        fsl_command = (
            "bet "
            + scan_file_name
            + " "
            + scan_file_base
            + "_bet"
            + constants.DATA_NIFTI_EXTENSION
            + " -m -Z -f 0.4 -n"
        )

        bind_string = f"{get_path_as_string(scan_folder)}:/data/"

        container_output = singularity_client.execute(
            (
                get_path_as_string(fsl_singularity_container)
                if isinstance(fsl_singularity_container, Path)
                else fsl_singularity_container
            ),
            fsl_command,
            stream=True,
            quiet=False,
            bind=bind_string,
            options=["--no-mount", "hostfs", "--pwd", "/data/"],
        )
        for line in container_output:
            print(line)

        bet_mask_file = scan_folder.joinpath(scan_file_base + "_bet_mask").with_suffix(
            constants.DATA_NIFTI_EXTENSION
        )

        bet_mask = sitk.ReadImage(bet_mask_file)
        scan = sitk.ReadImage(i_input_file)

        bet_mask.CopyInformation(scan)

        sitk.WriteImage(
            bet_mask,
            output_folder.joinpath(
                constants.BRAIN_MASK_NAME + scan_file_base
            ).with_suffix(constants.DATA_NIFTI_EXTENSION),
        )

        bet_mask_file.unlink()


def add_folders_for_brain_mask_at_each_step(input_dir: str | Path) -> None:
    """
    Add a folder for brain masks at each step of the preprocessing pipeline.

    Args:
        input_dir (str | Path): The directory containing the input data.
    """
    for i_patient_folder in Path(input_dir).iterdir():
        if i_patient_folder.is_dir():
            brain_masks_folder = i_patient_folder.joinpath("BRAIN_MASKS")
            original_folder = brain_masks_folder.joinpath("NIFTI")
            original_folder.mkdir(parents=True, exist_ok=True)
            registered_folder = brain_masks_folder.joinpath("REGISTERED")
            registered_folder.mkdir(parents=True, exist_ok=True)
            preprocessed_folder = brain_masks_folder.joinpath("FINAL")
            preprocessed_folder.mkdir(parents=True, exist_ok=True)
            for dir in brain_masks_folder.iterdir():
                if dir.is_file() and str(dir).endswith(constants.DATA_NIFTI_EXTENSION):
                    shutil.move(dir, registered_folder.joinpath(dir.name))


def crop_registered_scans_and_tumor_mask(
    input_dir: str | Path, mask_mni_dir: str | Path
) -> None:
    """
    Crop the registered brain scans and tumor masks to the MNI mask bounding box.

    Args:
        input_dir (str | Path): The directory containing the input data.
        mask_mni_dir (str | Path): The path to the MNI mask.
    """
    for i_patient_folder in Path(input_dir).iterdir():
        print(f"Now cropping patient: {i_patient_folder.name}", flush=True)
        if i_patient_folder.is_dir():
            registered_folder = i_patient_folder.joinpath("REGISTERED")
            cropped_folder = i_patient_folder.joinpath("CROPPED")
            cropped_folder.mkdir(parents=True, exist_ok=True)
            for i_modality in constants.SCAN_TYPES:
                i_input_file = registered_folder.joinpath(i_modality).with_suffix(
                    constants.DATA_NIFTI_EXTENSION
                )
                if i_input_file.exists():

                    cropped_scan, cropped_brain_mask = utils.crop_to_mask(
                        sitk.ReadImage(i_input_file),
                        sitk.Cast(sitk.ReadImage(mask_mni_dir), sitk.sitkUInt8),
                    )

                    normalized_scan = utils.normalize_intensity_with_mask(
                        cropped_scan, cropped_brain_mask
                    )

                    minimized_bg_scan = utils.mask_background_to_min(
                        normalized_scan, cropped_brain_mask
                    )
                    # Save preprocessed scan
                    sitk.WriteImage(
                        minimized_bg_scan,
                        cropped_folder.joinpath(
                            i_modality + constants.DATA_NIFTI_EXTENSION
                        ),
                    )

            # Collapsing labels of tumor mask
            current_patient_mask = registered_folder.joinpath(
                constants.MASK_FILE_NAME
            ).with_suffix(constants.DATA_NIFTI_EXTENSION)
            collapsed_tumor_mask = utils.collapse_mask_labels(
                sitk.Cast(sitk.ReadImage(current_patient_mask), sitk.sitkUInt8)
            )
            # Cropping tumor mask to bounding box
            cropped_tumor_mask, _ = utils.crop_to_mask(
                collapsed_tumor_mask,
                sitk.Cast(sitk.ReadImage(mask_mni_dir), sitk.sitkUInt8),
            )

            sitk.WriteImage(
                cropped_tumor_mask,
                cropped_folder.joinpath(constants.MASK_FILE_NAME).with_suffix(
                    constants.DATA_NIFTI_EXTENSION
                ),
            )


def crop_brain_masks_generated_from_registered_scans(
    input_dir: str | Path, mask_mni_dir: str | Path
) -> None:
    """
    Crop the brain masks generated from registered scans to the MNI mask.

    Args:
        input_dir (str | Path): The directory containing the input data.
        mask_mni_dir (str | Path): The path to the MNI mask.
    """
    for i_patient_folder in Path(input_dir).iterdir():
        if i_patient_folder.is_dir():
            print(f"Now cropping patient: {i_patient_folder.name}", flush=True)
            brain_masks_folder = i_patient_folder.joinpath("BRAIN_MASKS")
            brain_masks_from_registered = brain_masks_folder.joinpath("REGISTERED")
            cropped_folder = brain_masks_folder.joinpath("CROPPED_FROM_REGISTERED")
            cropped_folder.mkdir(parents=True, exist_ok=True)
            for i_modality in constants.SCAN_TYPES:
                i_input_file = brain_masks_from_registered.joinpath(
                    constants.BRAIN_MASK_NAME + i_modality
                ).with_suffix(constants.DATA_NIFTI_EXTENSION)
                if i_input_file.exists():
                    cropped_brain_mask, _ = utils.crop_to_mask(
                        sitk.Cast(sitk.ReadImage(i_input_file), sitk.sitkUInt8),
                        sitk.Cast(sitk.ReadImage(mask_mni_dir), sitk.sitkUInt8),
                    )
                    sitk.WriteImage(
                        cropped_brain_mask,
                        cropped_folder.joinpath(
                            i_modality + "_mask" + constants.DATA_NIFTI_EXTENSION
                        ),
                    )


def combine_brain_masks(
    input_dir: str | Path,
) -> None:
    """
    Combine the brain masks from all patients into a single mask.

    Args:
        input_dir (str | Path): The directory containing the input data.
    """
    for i_patient_folder in Path(input_dir).iterdir():
        if i_patient_folder.is_dir():
            cropped_folder = i_patient_folder.joinpath(
                "BRAIN_MASKS", "CROPPED_FROM_REGISTERED"
            )
            modalities_per_patient = []
            or_image_filter = sitk.OrImageFilter()
            for i_modality in constants.SCAN_TYPES:
                i_input_file = cropped_folder.joinpath(
                    i_modality + "_mask" + constants.DATA_NIFTI_EXTENSION
                )
                if not i_input_file.is_file():
                    raise FileNotFoundError(
                        f"Cannot combine brain masks; missing: {i_input_file}"
                    )
                current_mask = sitk.Cast(sitk.ReadImage(i_input_file), sitk.sitkUInt8)
                modalities_per_patient.append(current_mask)
            combined_mask_1 = or_image_filter.Execute(
                modalities_per_patient[0], modalities_per_patient[1]
            )
            combined_mask_2 = or_image_filter.Execute(
                combined_mask_1, modalities_per_patient[2]
            )
            final_combined_mask = or_image_filter.Execute(
                combined_mask_2, modalities_per_patient[3]
            )

            sitk.WriteImage(
                final_combined_mask,
                i_patient_folder.joinpath(
                    "BRAIN_MASKS",
                    "CROPPED_FROM_REGISTERED",
                    "combined_brain_mask.nii.gz",
                ),
            )


def clean_brain_masks(input_dir: str | Path) -> None:
    """
    Remove brain masks that are not needed for further analysis.

    Args:
        input_dir (str | Path): The directory containing the input data.
    """
    for i_patient_folder in Path(input_dir).iterdir():
        if i_patient_folder.is_dir():
            brain_masks_folder = i_patient_folder.joinpath("BRAIN_MASKS")
            for brain_dir in brain_masks_folder.iterdir():
                if brain_dir.is_dir():
                    print(len(list(brain_dir.iterdir())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and crop brain masks for patient folders."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Name of the configuration file.",
        metavar="configuration file",
        dest="config",
        type=str,
    )
    parser.add_argument(
        "--mni-mask",
        type=Path,
        help="Atlas brain mask on the same grid as the registered scans.",
    )
    args = parser.parse_args()

    this_script_dir = Path(__file__).parent.resolve()
    config_dir = this_script_dir.parent.resolve().joinpath("configs", args.config)
    config = configIO.Config(config_dir)

    input_dir = config.data_preprocess_dir
    mask_mni_dir = this_script_dir.joinpath(
        constants.MNI_ATLAS_FOLDER, constants.MNI_ATLAS_MASK
    ).as_posix()
    if args.mni_mask is not None:
        mask_mni_dir = str(args.mni_mask.expanduser().resolve(strict=True))
    if not Path(mask_mni_dir).is_file():
        parser.error(
            "Atlas mask is not bundled. Supply --mni-mask on the registered scan grid."
        )
    singularity_container_hdbet = config.container_path.joinpath(
        f"hdbet_{config.hdbet_version}.sif"
    )
    singularity_container_fslbet = config.container_path.joinpath(
        f"fsl_{config.fsl_version}.sif"
    )
    make_brain_masks(
        input_dir=input_dir,
        data_type="registered",
        singularity_container_hdbet_path=singularity_container_hdbet,
        singularity_container_fsl_bet_path=singularity_container_fslbet,
    )
    crop_brain_masks_generated_from_registered_scans(
        input_dir=input_dir,
        mask_mni_dir=mask_mni_dir,
    )
    combine_brain_masks(input_dir=input_dir)
