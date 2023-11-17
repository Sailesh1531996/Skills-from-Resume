import spacy
from spacy.lang.en import English
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL

# Load a pre-trained spaCy model with text classification capabilities
nlp = spacy.load("en_core_web_sm")

# Define the labels for classification
labels = ["Positive", "Negative", "Neutral"]

# Create a TextCategorizer object and add it to the spaCy pipeline
textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": DEFAULT_SINGLE_TEXTCAT_MODEL})

for label in labels:
    textcat.add_label(label)
nlp.add_pipe(textcat)

# Train the TextCategorizer on a labeled dataset of sentences
train_data = [("This is a positive sentence.", {"cats": {"Positive": 1, "Negative": 0, "Neutral": 0}}),
              ("This is a negative sentence.", {"cats": {"Positive": 0, "Negative": 1, "Neutral": 0}}),
              ("This is a neutral sentence.", {"cats": {"Positive": 0, "Negative": 0, "Neutral": 1}})]
for text, annotations in train_data:
    doc = nlp(text)
    for label, value in annotations['cats'].items():
        textcat = nlp.get_pipe('textcat')
        textcat.add_label(label)
        doc.cats[label] = value
    nlp.update([doc], losses={'textcat': textcat})

# Use the trained TextCategorizer to classify new sentences
test_data = ["This is a great day!", "I feel sad today.", "The weather is neither good nor bad."]
for text in test_data:
    doc = nlp(text)
    print(text)
    for label, score in doc.cats.items():
        print(f"{label}: {score}")
