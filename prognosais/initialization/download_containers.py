"""Download configured brain-extraction containers on any supported Linux host."""

import argparse
import shutil
import subprocess
from pathlib import Path

from prognosais.IO.config import Config


def main() -> None:
    """Download the explicit HD-BET and FSL tags from the configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--runtime", choices=("apptainer", "singularity"), required=True
    )
    args = parser.parse_args()
    runtime = shutil.which(args.runtime)
    if runtime is None:
        parser.error(f"Container runtime {args.runtime} is not on PATH.")
    config = Config(args.config.resolve(strict=True))
    destination = config.container_path
    if not config.config["containers"]["download_path"].strip():
        parser.error("Set containers.download_path to an explicit directory.")
    destination.mkdir(parents=True, exist_ok=True)
    for name, repository, version in (
        ("hdbet", "svdvoort/hdbet", config.hdbet_version),
        ("fsl", "fnndsc/fsl", config.fsl_version),
    ):
        if not version or version == "REPLACE_WITH_TAG" or "/" in version:
            parser.error(f"Set containers.{name}_version to a valid image tag.")
        target = destination / f"{name}_{version}.sif"
        if target.exists():
            print(f"Already present: {target}")
            continue
        subprocess.run(
            [runtime, "pull", str(target), f"docker://{repository}:{version}"],
            check=True,
        )
        print(target)


if __name__ == "__main__":
    main()
