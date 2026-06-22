## Exporting the fine-tuned TakeMeter model from Colab

After you have run fine-tuning in `colab/ai201_project3_takemeter_starter_tuned.ipynb` (Section 3) and see `✅ Fine-tuning complete`, run the cell below once to export the model and tokenizer and zip them for download.

Paste this into a new code cell at the end of Section 3 (or anywhere *after* `trainer.train()` has finished):

```python
# Export fine-tuned TakeMeter model + tokenizer
export_dir = "./takemeter-model"

# Save the best checkpoint selected by Trainer
trainer.save_model(export_dir)
tokenizer.save_pretrained(export_dir)

print(f"Saved model and tokenizer to {export_dir}")

# Optional: zip for easy download
import shutil
shutil.make_archive("takemeter-model", "zip", export_dir)
print("Created takemeter-model.zip")
```

Then, in the Colab **Files** pane:

1. Right‑click `takemeter-model.zip` and choose **Download**.
2. On your machine, unzip it into your project root so you have:
   - `takemeter-model/config.json`
   - `takemeter-model/pytorch_model.bin`
   - `takemeter-model/tokenizer.json` (and related tokenizer files)
3. Confirm that your repository now contains a `takemeter-model/` directory alongside `classify.py`.

The `classify.py` CLI will look for the model in `./takemeter-model` by default. If the directory is missing or incomplete, the script will print an error pointing back to this file.

