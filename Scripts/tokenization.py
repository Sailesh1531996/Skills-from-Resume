from nltk.stem import PorterStemmer
import nltk
import re
import csv

nltk.download('stopwords')

from nltk.corpus import stopwords


def token(sentence):
    result = []
    # Use regular expressions to remove symbols
    clean_sentence = re.sub(r'[^\w\s]', '', sentence)
    # Get the English stopwords from NLTK
    stop_words = set(stopwords.words('english'))
    # Tokenize the sentence
    tokens = nltk.word_tokenize(clean_sentence)
    # Remove the stopwords from the tokens
    filtered_tokens = [word for word in tokens if not word.lower() in stop_words]
    stemmer = PorterStemmer()

    for string in filtered_tokens:
        stemmed_word = stemmer.stem(string)
        result.append(stemmed_word)
        # print(result)

    return result

