from src.infer.predict import predict_all_tasks


def interactive_loop(encoder, multitask_model, id2label, device: str = "cpu") -> None:
    print("Interactive mode started. Write 'exit' to finish.")
    while True:
        text = input("text> ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Bye")
            break
        if not text:
            continue
        output = predict_all_tasks(text, encoder, multitask_model, id2label, device=device)
        print(output)
