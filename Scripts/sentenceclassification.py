import spacy
import classy_classification
import csv
from tokenization import token

def requirement(lines):
    data = {
    }
    with open('data/offers.txt', "r") as f:
        offers = f.read().splitlines()
    data["offers"] = offers

    with open('data/others.txt', "r") as f:
        others = f.read().splitlines()
    data["others"] = others

    with open('data/requirement.txt', "r") as f:
        requirement = f.read().splitlines()
    data["requirement"] = requirement

    nlp = spacy.blank("en")
    nlp.add_pipe(

        "text_categorizer",
        config={
            "data": data,
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "device": "cpu"
        }
    )
    sentence_model = spacy.blank("en")
    sentence_model.add_pipe("sentencizer")



    sentences = []
    for line in lines:
        line = line.strip()  # remove leading/trailing white space
        if line:  # ignore empty lines
            sentences.extend(line.split(". "))  # split line into sentences

    final_data = []
    for sentence in sentences:
        # print(sentence)
        doc = nlp(sentence)
        final_data.append({"sentence": doc.text, "cats": doc._.cats})

    intermediateData = ''
    result = []
    for item in final_data:
        if item["cats"]["requirement"] > .96:
             intermediateData =  intermediateData + "\n" +  item["sentence"].strip()
             #print (item["sentence"].strip())
             #print (item["cats"])
             #print ()

    return(intermediateData)






