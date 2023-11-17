import csv

header = ['Job Description', 'Requirements', 'Keywords', 'Cluster']
data=[]

with open('data/output.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
