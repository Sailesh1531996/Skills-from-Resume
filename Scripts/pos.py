import nltk

# sample text data
text = "Job Description: We are looking for support immediately! Are you looking for a part-time job (max. 20 hours/week, alternatively a mini-job is also possible)? Do you have an apprenticeship in the motor vehicle or two-wheeler sector? Or would you like to venture into this area?Then apply! We are looking for an employee with manual skills and technical understanding. In return, we offer you working in flat hierarchies and performance-related pay. We look forward to receiving your application by email or telephone!"

nltk.download('averaged_perceptron_tagger')
# use the word_tokenize() method from the nltk library to tokenize the text into words
words = nltk.word_tokenize(text)

# use the pos_tag() method from the nltk library to perform POS tagging on the words
pos_tags = nltk.pos_tag(words)

# loop through the tagged words and extract the nouns and verbs
nouns = []
verbs = []

for word, pos in pos_tags:
    if pos.startswith('N'):
        nouns.append(word)
    elif pos.startswith('V'):
        verbs.append(word)

# print the extracted nouns and verbs
print("Nouns:", nouns)
print("Verbs:", verbs)
