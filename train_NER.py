import argparse
import json
import numpy as np
import torch
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EvalPrediction
)
from evaluate import load

# Configuration
label2id = {"O": 0, "PRODUCT": 1}
id2label = {v: k for k, v in label2id.items()}
metric = load("seqeval")


def load_data(path):
    """Improved data loading with validation and chunking"""
    with open(path, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    samples = []
    entity_count = 0

    for page in pages:
        text = page['text'].strip()
        entities = page['entities']

        # Validate and filter entities
        valid_entities = []
        for ent in entities:
            if ent['end'] > len(text):
                continue
            ent_text = text[ent['start']:ent['end']]
            if not ent_text.strip() or len(ent_text) < 2:
                continue
            valid_entities.append(ent)
            entity_count += 1

        # Split text into chunks with overlap
        chunk_size = 128
        overlap = 25
        for i in range(0, len(text), chunk_size - overlap):
            chunk_start = max(0, i - overlap)
            chunk_end = min(len(text), i + chunk_size)
            chunk = text[chunk_start:chunk_end]

            chunk_entities = []
            for ent in valid_entities:
                if ent['start'] >= chunk_start and ent['end'] <= chunk_end:
                    new_ent = {
                        'start': ent['start'] - chunk_start,
                        'end': ent['end'] - chunk_start,
                        'label': ent['label']
                    }
                    if 0 <= new_ent['start'] < new_ent['end'] <= len(chunk):
                        chunk_entities.append(new_ent)

            # Simple data augmentation
            if chunk_entities and random.random() < 0.3:
                chunk = chunk.replace(" купить ", " приобрести ")
                chunk = chunk.replace(" цена ", " стоимость ")

            samples.append({
                'text': chunk,
                'entities': chunk_entities
            })

    print(f"Loaded {len(samples)} samples with {entity_count} valid entities")
    return Dataset.from_dict({
        'text': [s['text'] for s in samples],
        'entities': [s['entities'] for s in samples]
    })


def align_labels(examples, tokenizer):
    """Robust label alignment with token mapping"""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        return_offsets_mapping=True
    )

    labels = []
    for i, (text, ents) in enumerate(zip(examples['text'], examples['entities'])):
        text_len = len(text)
        label = np.zeros(len(tokenized['input_ids'][i]), dtype=int)

        for ent in ents:
            ent_start = max(0, min(ent['start'], text_len - 1))
            ent_end = max(ent_start + 1, min(ent['end'], text_len))

            # Find overlapping tokens
            for token_idx, (token_start, token_end) in enumerate(tokenized['offset_mapping'][i]):
                if token_start >= ent_end or token_end <= ent_start:
                    continue
                label[token_idx] = label2id[ent['label']]

        labels.append(label.tolist())

    tokenized.pop('offset_mapping')
    tokenized['labels'] = labels
    return tokenized


def compute_metrics(p: EvalPrediction):
    predictions, labels = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, lbl) if l != -100]
        for pred, lbl in zip(predictions, labels)
    ]

    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lbl) if l != -100]
        for pred, lbl in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 3.0], device=model.device)
        )
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main(args):
    # Device configuration
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Model initialization
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModelForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Data preparation
    dataset = load_data(args.data_dir)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenized_train = dataset["train"].map(
        lambda x: align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    tokenized_val = dataset["test"].map(
        lambda x: align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    # Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.05,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=device.type == 'cuda',
        gradient_accumulation_steps=2,
        report_to="none"
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Training and evaluation
    print("Starting training...")
    trainer.train()

    print("\nFinal evaluation:")
    results = trainer.evaluate()
    print(f"Validation F1: {results['eval_f1']:.3f}")
    print(f"Validation Precision: {results['eval_precision']:.3f}")
    print(f"Validation Recall: {results['eval_recall']:.3f}")

    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to annotations.json')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    args = parser.parse_args()
    main(args)