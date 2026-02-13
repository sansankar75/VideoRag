###  Its cleaned the data by consider the id for objects
import json

newData=[]
NewId=[]

with open("cleaned_detection.json","r") as j:

    t=json.load(j)
    
    ## Add new id in array
    for elements in t:
        for e in elements['objects']: 
            if e["id"] not in NewId:
                NewId.append(e['id'])

    #print(NewId)   
    print(len(NewId))   # len of id (count of all objects in video)
    
    ## Add elements into newData file based on id 
    for elements in t:
        for e in elements['objects']: 
            id=e['id']
            if id in NewId:
                #print(id)
                NewId.remove(id)
                #print(NewId)
                newData.append({"objects": [{
                "id": e['id'],
                "label": e['label'],
                "confidence":e['confidence'] 
            }]})

            

# Save json file 
with open("newData.json",'w') as newFile:
    json.dump(newData,newFile)




## count the objects in cleaned frame
newDict={}
c=0
with open("newData.json","r") as j:
    t=json.load(j)
    
    for elements in t:
        for e in elements['objects']: 
    
            if e['label'] not in newDict:
                newDict.update({e['label']:1})
            else:
                newDict[e['label']]+=1
            c+=1

for i in newDict:
    print(i,newDict[i])


        
    
