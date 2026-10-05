import argparse
import shutil
from pathlib import Path

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.preprocessing.utils import convert_dicoms_to_niftis

parser = argparse.ArgumentParser(
    description="Preprocess data for the prognosais pipeline"
)
parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Name of the configuration file",
    metavar="configuration file",
    dest="config",
    type=str,
)

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
config_dir = this_script_dir.parent.resolve().joinpath("configs", args.config)
print(
    f"This job and script {Path(__file__).resolve()} was run with the following config file: {config_dir}",
    flush=True,
)

config = configIO.Config(config_dir)
data_dir = Path(config.data_preprocess_dir)
failed_dir = data_dir.parent / constants.FAILED_PATIENTS_FOLDER
failed_dir.mkdir(parents=True, exist_ok=True)
patients = [i_patient for i_patient in sorted(data_dir.iterdir()) if i_patient.is_dir()]

if __name__ == "__main__":
    if not patients:
        raise ValueError(f"No patient directories found in {data_dir}")
    failed_cases = []
    for i_patient in patients:
        cur_dir = i_patient.name
        try:
            convert_dicoms_to_niftis(i_patient)
        except Exception as e:
            print(f"Dicom-2-nifti conversion failed for patient {cur_dir}: {e}")
            print(f"Moving {cur_dir} to 'failed_patients' folder")
            print()
            shutil.move(str(i_patient), str(failed_dir / cur_dir))
            failed_cases.append(cur_dir)
    if failed_cases:
        raise RuntimeError(
            f"DICOM conversion failed for {len(failed_cases)} case(s): "
            f"{', '.join(failed_cases)}. Inspect {failed_dir}."
        )
