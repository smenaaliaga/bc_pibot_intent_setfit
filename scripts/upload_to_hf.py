import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hub.upload import upload_artifacts_to_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload multitask artifacts to Hugging Face Hub")
    parser.add_argument("--artifact-dir", default="models/artifacts")
    parser.add_argument("--repo-id", required=True, help="e.g. username/bc-pibot-intent-setfit")
    parser.add_argument("--hf-token", help="Optional. If omitted, uses token from `huggingface-cli login`.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--clean-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing files in HF repo before uploading this package.",
    )
    parser.add_argument("--commit-message", default="Upload multitask artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_artifacts_to_hf(
        local_dir=Path(args.artifact_dir),
        repo_id=args.repo_id,
        token=args.hf_token,
        private=args.private,
        clean_repo=args.clean_repo,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.artifact_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
