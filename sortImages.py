#aionkov

import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# open open category to id file
with open('val2019.json') as f:
    values = json.load(f)

#open id to images file
with open('train2019.json') as f:
    images = json.load(f)


annotation_images = pd.DataFrame.from_dict(images["annotations"])
catagory_images = pd.DataFrame.from_dict(images["categories"])
images_id = pd.DataFrame.from_dict(images["images"])

train_data = pd.DataFrame({'ImageId':None,'Class':None,'Family':None,'Genus':None,'Kingdom':None,'Name':None,'Order':None,'Phylum':None}, index = [0])
    
index = 0;
for t in annotation_images.category_id:
    current_info = pd.DataFrame(catagory_images.loc[t]).T
    
    train_data.loc[index] = [index, current_info["class"], current_info["family"], current_info["genus"], current_info["kingdom"], current_info["name"], current_info["order"], current_info["phylum"]]
    print(index)
    index += 1
    
labelencoder = LabelEncoder()
train_data[:, 1] = train_data.fit_transform(x[:, 1])

lower_train_data = pd.DataFrame({'FileName': images_id['file_name'],'ImageId': annotation_images['id'], 'CatagoryID':annotation_images['category_id']})

lower_train_data = lower_train_data.sort_values(by=['CatagoryID'])

annotations = data["annotations"]
#get path of file
path = os.path.dirname(os.path.abspath(__file__))

for x in annotations:
	#make the dir
	if not os.path.exists(x):
    	os.makedirs(path + "/" + x)
    #for every image id
    for y in annotations[x]:
    	#get image path
    	imagename = images["images"][y]
    	#rename the path to our new directory x
    	print("RENAME: " + imagename + ".jpg  | to: " path + "/" + x + "/" + y + ".jpg")
    	os.rename(imagename + ".jpg", path + "/" + x + "/" + y + ".jpg")

print("fuck yeah")