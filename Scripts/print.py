import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the BERT model and tokenizer
model_name = 'path/directory'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(model_name)

# Example sentences
sentence1 = "A digital fingerprint is an almost unambiguous identification of data as a result of a hash function."
sentence2 = "term collision proof means transaction independent does have conflicts"
# Tokenize and encode the sentences
tokens1 = tokenizer.encode_plus(sentence1, truncation=True, padding=True, return_tensors='pt')


tokens2 = tokenizer.encode_plus(sentence2, truncation=True, padding=True, return_tensors='pt')

# Generate sentence embeddings
with torch.no_grad():
    outputs1 = model(tokens1['input_ids'], attention_mask= tokens1['attention_mask'])
    sentence_embeddings1 = outputs1.last_hidden_state[:, 0, :].numpy()

    outputs2 = model(tokens2['input_ids'], attention_mask=tokens2['attention_mask'])
    sentence_embeddings2 = outputs2.last_hidden_state[:, 0, :].numpy()

# Calculate cosine similarity
similarity = cosine_similarity(sentence_embeddings1, sentence_embeddings2)
print(f"Cosine Similarity: {similarity[0][0]}")
