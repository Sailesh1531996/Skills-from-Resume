from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification, Trainer,TrainingArguments
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

print ("Hello")
data = {
    'sentence1': [
        "a digital fingerprint is an almost unambiguous identification of data as a result of a hash function",
        "it consists of a header and data related to transactions that take place within a specified period of time",
        "a hash function is called collision-proof when it is very difficult (or impossible) to find two or more different data for which it gives the identical hash value",
        "computer protocol that determines interactions between actors in any type of blockchain",
        "they support a decentralized communication model in which each node on the network has the same capabilities and can establish communication with all other nodes"
    ],
    'sentence2': [
        "digital fingerprint identifies person uniquely cannot same two people",
        "transaction block consists of information transaction history blockchain",
        "term collision proof means transaction independent does have conflicts",
        "smart contract protocol",
        "peer peer system does have centralized system more computational power"
    ],
    'similarity_score': [0.11, 0.18, 0.05, 0.12, 0.06]


}

print(data['sentence1'])

# Define the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)


# Tokenize the input sentences
def tokenize_fn(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True)


# Define the training arguments
training_args = TrainingArguments(

    output_dir='data/result.txt',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    evaluation_strategy='epoch'
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=tokenize_fn
)
# Fine-tune the BERT model
trainer.train()

trainer.save_model('data/result.txt')