import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.config.schema import DataConfig, TrainConfig
from src.data.datasets import split_by_task
from src.data.io_csv import build_label_maps, load_task_data
from src.hub.upload import upload_artifacts_to_hf
from src.infer.interactive import interactive_loop
from src.infer.predict import predict_all_tasks
from src.infer.qa_console import qa_loop
from src.model.encoder import SharedEncoder
from src.model.multitask_model import MultiTaskClassifier
from src.serialization.artifacts import load_artifacts, save_artifacts
from src.train.trainer import MultiTaskTrainer, build_datasets_from_frames
from src.utils.seed import seed_everything


def _build_data_config_from_args(args) -> DataConfig:
    return DataConfig(
        data_dir=Path(args.data_dir),
        macro_file=args.macro_file,
        intent_file=args.intent_file,
        context_file=args.context_file,
        text_col=args.text_col,
        label_col=args.label_col,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )


def _resolve_text_inputs(args) -> List[str]:
    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.texts_file:
        text_file = Path(args.texts_file)
        if not text_file.exists():
            raise FileNotFoundError(f"texts file not found: {text_file}")
        file_texts = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts.extend(file_texts)
    if not texts:
        raise ValueError("Provide at least one --text or a valid --texts-file")
    return texts


def train_command(args) -> None:
    data_config = _build_data_config_from_args(args)
    train_config = TrainConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        task_weight_macro=args.task_weight_macro,
        task_weight_intent=args.task_weight_intent,
        task_weight_context=args.task_weight_context,
        device=args.device,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    seed_everything(train_config.seed)

    task_data = load_task_data(data_config)
    label_maps = {}
    for task_name, frame in task_data.items():
        label2id, _ = build_label_maps(frame, data_config.label_col)
        label_maps[task_name] = label2id

    splits = split_by_task(task_data, data_config)
    for task_name, split in splits.items():
        print(
            f"[{task_name}] split sizes -> "
            f"train={len(split.train)} | val={len(split.val)} | test={len(split.test)}"
        )

    train_datasets, val_datasets, test_datasets = build_datasets_from_frames(
        split_frames=splits,
        label_maps=label_maps,
        text_col=data_config.text_col,
        label_col=data_config.label_col,
    )

    encoder = SharedEncoder(
        model_name=train_config.model_name,
        device=train_config.device,
        use_lora=train_config.use_lora,
        lora_r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
    )
    embedding_dim = encoder.encode(["_probe_"], convert_to_tensor=True).shape[-1]
    num_classes_by_task = {task: len(mapping) for task, mapping in label_maps.items()}
    multitask_model = MultiTaskClassifier(embedding_dim=embedding_dim, num_classes_by_task=num_classes_by_task)

    trainer = MultiTaskTrainer(encoder=encoder, model=multitask_model, train_config=train_config)
    trainer.train(train_datasets=train_datasets, val_datasets=val_datasets)

    final_val_metrics = trainer.evaluate(val_datasets)
    print(f"Final validation metrics: {final_val_metrics}")

    has_test = any(len(dataset) > 0 for dataset in test_datasets.values())
    if has_test:
        final_test_metrics = trainer.evaluate(test_datasets)
        print(f"Final test metrics: {final_test_metrics}")
    else:
        print("Final test metrics: skipped (test_size=0)")

    save_artifacts(
        output_dir=train_config.output_dir,
        encoder=encoder,
        multitask_model=multitask_model,
        label2id=label_maps,
        train_config=asdict(train_config),
    )
    print(f"Artifacts saved at: {train_config.output_dir}")


def evaluate_command(args) -> None:
    data_config = _build_data_config_from_args(args)
    _, multitask_model, label2id, _ = load_artifacts(Path(args.artifact_dir), device=args.device)
    encoder = SharedEncoder.load(str(Path(args.artifact_dir) / "encoder"), device=args.device)

    task_data = load_task_data(data_config)
    splits = split_by_task(task_data, data_config)
    _, val_datasets, test_datasets = build_datasets_from_frames(
        split_frames=splits,
        label_maps=label2id,
        text_col=data_config.text_col,
        label_col=data_config.label_col,
    )

    trainer = MultiTaskTrainer(
        encoder=encoder,
        model=multitask_model,
        train_config=TrainConfig(device=args.device),
    )
    if args.split == "val":
        metrics = trainer.evaluate(val_datasets)
    else:
        metrics = trainer.evaluate(test_datasets)
    print(metrics)


def test_command(args) -> None:
    encoder, multitask_model, _, id2label = load_artifacts(Path(args.artifact_dir), device=args.device)
    texts = _resolve_text_inputs(args)
    for index, text in enumerate(texts, start=1):
        output = predict_all_tasks(
            text=text,
            encoder=encoder,
            multitask_model=multitask_model,
            id2label=id2label,
            device=args.device,
        )
        print({"input_index": index, "text": text, "predictions": output})


def interactive_command(args) -> None:
    encoder, multitask_model, _, id2label = load_artifacts(Path(args.artifact_dir), device=args.device)
    interactive_loop(encoder, multitask_model, id2label, device=args.device)


def qa_command(args) -> None:
    encoder, multitask_model, _, id2label = load_artifacts(Path(args.artifact_dir), device=args.device)
    qa_loop(
        encoder=encoder,
        multitask_model=multitask_model,
        id2label=id2label,
        device=args.device,
        top_k=args.top_k,
    )


def upload_command(args) -> None:
    upload_artifacts_to_hf(
        local_dir=Path(args.artifact_dir),
        repo_id=args.repo_id,
        token=args.hf_token,
        private=args.private,
        clean_repo=args.clean_repo,
        commit_message=args.commit_message,
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-task SetFit style training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--macro-file", default="dataset_macro.csv")
    train_parser.add_argument("--intent-file", default="dataset_intent.csv")
    train_parser.add_argument("--context-file", default="dataset_context.csv")
    train_parser.add_argument("--text-col", default="text")
    train_parser.add_argument("--label-col", default="label")
    train_parser.add_argument("--output-dir", default="models/artifacts")
    train_parser.add_argument("--model-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--max-length", type=int, default=128)
    train_parser.add_argument("--lr-encoder", type=float, default=2e-5)
    train_parser.add_argument("--lr-heads", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--task-weight-macro", type=float, default=1.0)
    train_parser.add_argument("--task-weight-intent", type=float, default=1.0)
    train_parser.add_argument("--task-weight-context", type=float, default=1.0)
    train_parser.add_argument("--val-size", type=float, default=0.2)
    train_parser.add_argument("--test-size", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--use-lora", action="store_true", default=True, help="Enable LoRA for encoder fine-tuning")
    train_parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling")
    train_parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    train_parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0=disabled)")
    train_parser.add_argument("--min-delta", type=float, default=0.001, help="Min F1 improvement to reset patience")
    train_parser.set_defaults(func=train_command)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--artifact-dir", default="models/artifacts")
    evaluate_parser.add_argument("--data-dir", default="data")
    evaluate_parser.add_argument("--macro-file", default="dataset_macro.csv")
    evaluate_parser.add_argument("--intent-file", default="dataset_intent.csv")
    evaluate_parser.add_argument("--context-file", default="dataset_context.csv")
    evaluate_parser.add_argument("--text-col", default="text")
    evaluate_parser.add_argument("--label-col", default="label")
    evaluate_parser.add_argument("--val-size", type=float, default=0.2)
    evaluate_parser.add_argument("--test-size", type=float, default=0.1)
    evaluate_parser.add_argument("--split", choices=["val", "test"], default="val")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.set_defaults(func=evaluate_command)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--artifact-dir", default="models/artifacts")
    test_parser.add_argument("--text", action="append", help="Repeat --text to send multiple inputs")
    test_parser.add_argument("--texts-file", help="UTF-8 file with one text per line")
    test_parser.add_argument("--device", default="cpu")
    test_parser.set_defaults(func=test_command)

    interactive_parser = subparsers.add_parser("interactive")
    interactive_parser.add_argument("--artifact-dir", default="models/artifacts")
    interactive_parser.add_argument("--device", default="cpu")
    interactive_parser.set_defaults(func=interactive_command)

    qa_parser = subparsers.add_parser("qa")
    qa_parser.add_argument("--artifact-dir", default="models/artifacts")
    qa_parser.add_argument("--device", default="cpu")
    qa_parser.add_argument("--top-k", type=int, default=3)
    qa_parser.set_defaults(func=qa_command)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--artifact-dir", default="models/artifacts")
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--hf-token")
    upload_parser.add_argument("--private", action="store_true")
    upload_parser.add_argument(
        "--clean-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing files in HF repo before uploading this package.",
    )
    upload_parser.add_argument("--commit-message", default="Upload multitask artifacts")
    upload_parser.set_defaults(func=upload_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
