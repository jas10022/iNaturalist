#aionkov

import json
import os

# open open category to id file
with open('val2019.json') as f:
    values = json.load(f)

#open id to images file
with open('train2019.json') as f:
    images = json.load(f)

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