with open ("data/input.txt", "r") as f:
    lines = f.readlines()

sentences = []
for line in lines:
    line = line.strip()  # remove leading/trailing white space
    if line:  # ignore empty lines
        sentences.extend(line.split(". "))  # split line into sentences

final_data = []
for sentence in sentences :
    print(sentence)