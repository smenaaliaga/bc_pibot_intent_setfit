import json
from pathlib import Path


def test_artifact_file_names_smoke(tmp_path: Path):
    files = ["label2id.json", "id2label.json", "train_config.json"]
    for name in files:
        path = tmp_path / name
        path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        assert path.exists()
