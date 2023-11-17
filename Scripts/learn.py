import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_sm")

# Define the labels for the entities we want to recognize
labels = ["SKILL"]

# Get the training data
TRAIN_DATA = [("Must have experience with Python, SQL, and data analysis.", {"entities": [(23, 29, "SKILL"), (31, 34, "SKILL"), (39, 52, "SKILL")]}),
              ("We are looking for someone with expertise in machine learning.", {"entities": [(33, 46, "SKILL")]}),
              ("Candidate should have a strong background in software development.", {"entities": [(35, 52, "SKILL")]}),
              # Add more training examples here
              ]

# Add the labels to the NER pipeline
for label in labels:
    nlp.entity.add_label(label)

# Train the NER model
n_iter = 100
batch_size = 16

# Start the training
with nlp.disable_pipes("tagger", "parser"):
    # Only train the NER component
    ner = nlp.get_pipe("ner")
    optimizer = ner.begin_training()

    # Loop through the training data in batches
    for i in range(n_iter):
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(batch_size, max_batch_size=batch_size*3))

        # Update the model with each batch
        for batch in batches:
            texts, annotations = zip(*batch)
            ner.update(texts, annotations, sgd=optimizer, losses=losses)

        print("Iteration {} Losses: {}".format(i+1, losses))
