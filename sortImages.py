#aionkov

import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
    
scaled_features = catagory_images.copy()
col_names = ['class', 'family', 'genus','kingdom','name','order','phylum']
features = catagory_images[col_names]
for i in range(0,7):
    labelencoder = LabelEncoder()
    features.values[:,i] = labelencoder.fit_transform(features.values[:,i])

scaled_features[col_names] = features

index = 0;
for t in annotation_images.category_id:
    current_info = pd.DataFrame(scaled_features.loc[t]).T
    
    train_data.loc[index] = [index, current_info["class"].values[0], current_info["family"].values[0], current_info["genus"].values[0], current_info["kingdom"].values[0], current_info["name"].values[0], current_info["order"].values[0], current_info["phylum"].values[0]]
    print(index)
    index += 1

train_data = train_data.assign(File_Name=images_id['file_name'].values)

train_data.to_csv(r'/Users/jas10022/Documents/GitHub/iNaturalist/upperNN .csv', index=False)

lower_train_data = pd.DataFrame({'FileName': images_id['file_name'],'ImageId': annotation_images['id'], 'CatagoryID':annotation_images['category_id']})

lower_train_data = lower_train_data.sort_values(by=['CatagoryID'])

lower_train_data.set_index(keys=['CatagoryID'], drop=False,inplace=True)
# get a list of names
numbers=lower_train_data['CatagoryID'].unique().tolist()
# now we can perform a lookup on a 'view' of the dataframe

for number in numbers:
    data = lower_train_data.loc[lower_train_data['CatagoryID']==number]

    test = lower_train_data.sample(frac=0.0001)
    
    data = pd.concat([data,test])

    data.to_csv(r'/Users/jas10022/Documents/GitHub/iNaturalist/' + str(number) + '.csv', index=False)


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