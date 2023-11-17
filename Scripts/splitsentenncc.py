with open('data/harry_potter_cleaned.txt', 'r') as f:
    lines = f.readlines()

sentences = []
for line in lines:
    line = line.strip()  # remove leading/trailing white space
    if line:  # ignore empty lines
        sentences.extend(line.split(". "))  # split line into sentences

for sentence in sentences :
    print(sentence)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")