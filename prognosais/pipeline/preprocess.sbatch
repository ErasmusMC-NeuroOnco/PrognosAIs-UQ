#!/usr/bin/env bash
#SBATCH --job-name=prognosais-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=prognosais-preprocess-%j.out

set -euo pipefail

usage() {
    echo "Usage: sbatch [resource options] prognosais/pipeline/preprocess.sbatch {dicom|preprocess|brain-masks} CONFIG.yml [--mni-mask MASK.nii.gz]" >&2
    echo "Submit from the repository root; CONFIG.yml must be under prognosais/configs/." >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage
    exit 2
fi

stage=$1
config_name=$2
shift 2

case "$stage" in
    dicom)
        module_name=prognosais.preprocessing.dicom2nifti
        ;;
    preprocess)
        module_name=prognosais.preprocessing.preprocess
        ;;
    brain-masks)
        module_name=prognosais.preprocessing.extract_brain_masks
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ "$stage" != "brain-masks" && $# -gt 0 ]]; then
    usage
    exit 2
fi
if [[ "$stage" == "brain-masks" && $# -gt 0 && "${1:-}" != "--mni-mask" ]]; then
    usage
    exit 2
fi

# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked. A project
# path may be supplied explicitly when a site's submission setup changes it.
cd "${PROGNOSAIS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "$config_name" == /* || ! -f "prognosais/configs/$config_name" ]]; then
    echo "Config must be relative to prognosais/configs/: $config_name" >&2
    exit 2
fi
if [[ ! -f .venv/bin/activate ]]; then
    echo "Missing .venv; follow the README installation instructions first." >&2
    exit 2
fi

source .venv/bin/activate
export MPLBACKEND=Agg
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec python -m "$module_name" --config "$config_name" "$@"
