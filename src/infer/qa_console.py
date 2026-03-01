from src.infer.predict import predict_all_tasks_detailed


def _print_task(task_name: str, task_output: dict) -> None:
    label = str(task_output["label"])
    score = float(task_output["score"])
    print(f"{task_name}: {label} ({score:.3f})")
    top_k = task_output.get("top_k", [])
    if not top_k:
        return
    for rank, candidate in enumerate(top_k, start=1):
        candidate_label = str(candidate["label"])
        candidate_score = float(candidate["score"])
        print(f"  {rank}. {candidate_label} ({candidate_score:.3f})")


def qa_loop(encoder, multitask_model, id2label, device: str = "cpu", top_k: int = 3) -> None:
    print("QA mode started. Type text and press enter. Write 'exit' to finish.")
    while True:
        text = input("qa> ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Bye")
            break
        if not text:
            continue

        output = predict_all_tasks_detailed(
            text=text,
            encoder=encoder,
            multitask_model=multitask_model,
            id2label=id2label,
            device=device,
            top_k=top_k,
        )

        print("-" * 48)
        print(f"text: {text}")
        _print_task("macro", output["macro"])
        _print_task("intent", output["intent"])
        _print_task("context", output["context"])
