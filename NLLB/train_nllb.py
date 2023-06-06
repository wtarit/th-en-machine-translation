from datasets import load_from_disk
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import numpy as np
import evaluate
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--report_to", type=str, default="none")
parser.add_argument("--dataset", type=str)
parser.add_argument("--save_steps", type=int, default=10000)

args, _ = parser.parse_known_args()

checkpoint = "facebook/nllb-200-distilled-600M"
tokenizer = NllbTokenizerFast.from_pretrained(
    checkpoint, src_lang="tha_Thai", tgt_lang="eng_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
metric = evaluate.load("sacrebleu")
tokenized_sentence = load_from_disk(args.dataset, keep_in_memory=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


transformers.logging.set_verbosity_info()

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    fp16_full_eval=True,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=args.epochs,
    predict_with_generate=True,
    save_steps=args.save_steps,
    fp16=True,
    push_to_hub=False,
    report_to=args.report_to,
    run_name=args.model_name,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_sentence["train"],
    eval_dataset=tokenized_sentence["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{args.output_dir}/final_checkpoint")
