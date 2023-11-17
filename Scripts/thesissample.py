import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics.pairwise import cosine_similarity
# Step 1: Prepare your data (example data)
sentences1 = [
        "a digital fingerprint is an almost unambiguous identification of data as a result of a hash function",
        "it consists of a header and data related to transactions that take place within a specified period of time",
        "a hash function is called collision-proof when it is very difficult (or impossible) to find two or more different data for which it gives the identical hash value",
        "computer protocol that determines interactions between actors in any type of blockchain",
        "they support a decentralized communication model in which each node on the network has the same capabilities and can establish communication with all other nodes"
    ]

sentences2 = [
        "digital fingerprint identifies person uniquely cannot same two people",
        "transaction block consists of information transaction history blockchain",
        "term collision proof means transaction independent does have conflicts",
        "smart contract protocol",
        "peer peer system does have centralized system more computational power"
    ]

# Corresponding similarity scores for each pair of sentences
similarity_scores = [0.11, 0.18, 0.05, 0.12, 0.06]

# Step 2: Preprocess the data (not required in this example)

# Step 3: Tokenization and encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

inputs = tokenizer(sentences1, sentences2, truncation=True, padding=True, return_tensors='pt')

labels = torch.tensor(similarity_scores)


# Step 4: Define a custom dataset
class SimilarityDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item


dataset = SimilarityDataset(inputs, labels)

# Step 5: Define the BERT model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, from_tf=True)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Step 6: Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader)}")


model.save_pretrained('path/directory')


