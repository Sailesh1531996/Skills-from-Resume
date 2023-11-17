import csv
from sentenceclassification import requirement
from tokenization import token
from clustering import clusteringData

with open("data/input.txt", "r") as f:
    lines = f.readlines()

jobs =[]
totaljobs=[]
for line in lines :
    if(line!="-----\n"):
        jobs.append(line)
    else:
        totaljobs.append(jobs)
        jobs=[]
totaljobs.append(jobs)

resultData = []
totalRequirements = []
totalToken =[]

for jobs in totaljobs:
    data = []
    result =requirement(jobs)
    resultToken = token(result)
    stringToken = ', '.join(resultToken)
    totalRequirements.append(result)
    totalToken.append(stringToken)



#print(resultData)

clusters = clusteringData(totalRequirements)
clustersKeyword = clusteringData(totalToken)

for i in range(len(totaljobs)):
    data = []
    data.append(totaljobs[i])
    data.append(totalRequirements[i])
    data.append(totalToken[i])
    data.append(clusters[i])
    data.append(clustersKeyword[i])
    resultData.append(data)




header = ['Job Description', 'Requirements', 'Keywords', 'Cluster based on Requirements' , 'Cluster based on Keywords']
with open('data/output.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(resultData)






