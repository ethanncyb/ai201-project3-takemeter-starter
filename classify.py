import argparse
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "takemeter-model")

LABEL_MAP = {
    "analysis": 0,
    "hot_take": 1,
    "reaction": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_model(model_dir: str = DEFAULT_MODEL_DIR):
    if not os.path.isdir(model_dir):
        sys.stderr.write(
            f"Error: model directory '{model_dir}' not found.\n"
            "Export your fine-tuned model from Colab using MODEL_EXPORT.md, then unzip "
            "it so that 'takemeter-model/' lives next to this script.\n"
        )
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"Error: failed to load model from '{model_dir}': {exc}\n"
            "Make sure you ran the export cell in MODEL_EXPORT.md and that all files "
            "are present.\n"
        )
        sys.exit(1)


def classify_text(text: str, tokenizer, model) -> tuple[str, float]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    pred_label = ID_TO_LABEL.get(pred_id, str(pred_id))
    confidence = float(probs[pred_id].item())
    return pred_label, confidence


def run_one_shot(text: str, model_dir: str) -> None:
    tokenizer, model = load_model(model_dir)
    label, conf = classify_text(text, tokenizer, model)
    print(f"Predicted: {label}  (confidence: {conf:.2f})")


def run_repl(model_dir: str) -> None:
    tokenizer, model = load_model(model_dir)
    print("TakeMeter CLI — enter a post to classify (blank line to quit).")
    while True:
        try:
            line = input("Post> ").strip()
        except EOFError:
            print()
            break

        if not line:
            break

        label, conf = classify_text(line, tokenizer, model)
        print(f"Predicted: {label}  (confidence: {conf:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a post with the fine-tuned TakeMeter model and print "
            "label + confidence."
        ),
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Post text to classify. If omitted, launches interactive mode.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing the fine-tuned model (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args()

    if args.text:
        run_one_shot(args.text, args.model_dir)
    else:
        run_repl(args.model_dir)


if __name__ == "__main__":
    main()

