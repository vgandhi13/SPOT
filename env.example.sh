# Template for local/env.sh (git-ignored). Copy it and fill in your own values:
#   mkdir -p local && cp env.example.sh local/env.sh
# Scripts under local/ source local/env.sh; code under eval/ takes everything as arguments.

# bin/ directory of the Python environment (venv or conda) with eval/glimpse/requirements.txt.
export SPOT_ENV_BIN="/path/to/env/bin"
# Modules to load before running (e.g. one providing ffprobe); leave empty if not needed.
export SPOT_MODULES="ffmpeg"

# Owner-only (chmod 600) file containing the OpenRouter API key.
export OPENROUTER_API_KEY_FILE="/path/to/private/openrouter_api_key"
# Owner-only file containing a Hugging Face token with write access (for dataset uploads).
export HF_TOKEN_FILE="/path/to/private/hf_token"

# Video-Holmes checkout: contains train_Video-Holmes.json, test_Video-Holmes.json, videos_cropped/.
export HOLMES_BENCHMARK="/path/to/Video-Holmes/Benchmark"

# Fast storage for raw run directories (the runner refuses /home and /project).
export SPOT_RUN_ROOT="/path/to/scratch/spot-runs"

# Slurm account/partition, read by sbatch itself (source this file before sbatch).
export SBATCH_ACCOUNT="your_account"
export SBATCH_PARTITION="cpu"
