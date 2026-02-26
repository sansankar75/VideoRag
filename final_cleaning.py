###    its cleaned the data by consider the id for objects
import json

newData={}

with open("newData.json","r") as j:
    t=json.load(j)
    
    for elements in t:
        for e in elements['objects']: 
            if e['label'] not in newData:
                newData.update({e['label']:1})
            else:
                newData[e['label']]+=1

for i in newData:
    print(i,newData[i])


        
    
