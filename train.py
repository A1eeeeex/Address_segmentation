from transformers import AutoTokenizer, BertForTokenClassification
import datasets

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertForTokenClassification.from_pretrained("bert-base-chinese")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    label2id = {
    "O":0,
    "T-B":1,
    "T-I":2,
    "P-B":3,
    "P-I":4,
    "A1-B":5,
    "A1-I":6,
    "A2-B":7,
    "A2-I":8,
    "A3-B":9,
    "A3-I":10,
    "A4-B":11,
    "A4-I":12
    }
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
import evaluate

seqeval = evaluate.load("seqeval")

import numpy as np

#labels = [label_list[i] for i in example[f"ner_tags"]]


def compute_metrics(p):
    label_list = [
        "O",
        "T-B",
        "T-I",
        "P-B",
        "P-I",
        "A1-B",
        "A1-I",
        "A2-B",
        "A2-I",
        "A3-B",
        "A3-I",
        "A4-B",
        "A4-I"
        ]
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    id2label = {
        0: "O",
        1: "T-B",
        2: "T-I",
        3: "P-B",
        4: "P-I",
        5: "A1-B",
        6: "A1-I",
        7: "A2-B",
        8: "A2-I",
        9: "A3-B",
        10: "A3-I",
        11: "A4-B",
        12: "A4-I",
    }
    label2id = {
        "O":0,
        "T-B":1,
        "T-I":2,
        "P-B":3,
        "P-I":4,
        "A1-B":5,
        "A1-I":6,
        "A2-B":7,
        "A2-I":8,
        "A3-B":9,
        "A3-I":10,
        "A4-B":11,
        "A4-I":12
    }
    label_list = [
        "O",
        "T-B",
        "T-I",
        "P-B",
        "P-I",
        "A1-B",
        "A1-I",
        "A2-B",
        "A2-I",
        "A3-B",
        "A3-I",
        "A4-B",
        "A4-I"
        ]
    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        "E:\工作\\transformer-model\\bert-base-chinese", num_labels=13, 
        id2label=id2label, label2id=label2id #use_auth_token=access_token
    )
    train = datasets.dataset_dict.DatasetDict.from_json('data\\train.json')
    tokenized_train = train.map(tokenize_and_align_labels, batched=True)
    dev = datasets.dataset_dict.DatasetDict.from_json('data\\dev.json')
    tokenized_dev = dev.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        #use_auth_token=access_token
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()