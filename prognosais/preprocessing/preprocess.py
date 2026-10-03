import argparse
from pathlib import Path

import prognosais.IO.config as configIO
from prognosais.preprocessing.Preprocessor import Preprocessor

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

if __name__ == "__main__":
    # Initialize Preprocessor
    preprocessor = Preprocessor(
        data_dir,
        this_script_dir,
        config.mask_origin_file_path,
        config.preprocess_registered_data,
        config.bias_field_correct_data,
    )

    # Run Preprocessor
    preprocessor.process_all_patients()
