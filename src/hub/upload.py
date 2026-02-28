from pathlib import Path
import shutil
import tempfile
from typing import Optional

from huggingface_hub import HfApi, upload_folder


def upload_artifacts_to_hf(
    local_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    clean_repo: bool = True,
    commit_message: str = "Upload multitask artifacts",
) -> None:
    local_dir = Path(local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {local_dir}")

    readme_file = Path("scripts/README.md")
    if not readme_file.exists() or not readme_file.is_file():
        raise FileNotFoundError(f"README file not found: {readme_file}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hf_upload_") as tmp_dir:
        staging_dir = Path(tmp_dir)
        shutil.copytree(local_dir, staging_dir, dirs_exist_ok=True)
        shutil.copy2(readme_file, staging_dir / "README.md")

        delete_patterns = ["*"] if clean_repo else None
        upload_folder(
            repo_id=repo_id,
            folder_path=str(staging_dir),
            token=token,
            repo_type="model",
            commit_message=commit_message,
            delete_patterns=delete_patterns,
        )
