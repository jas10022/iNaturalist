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
col_names = ['class', 'family','kingdom','order','phylum']
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

ordered_features = scaled_features.sort_values(['class', 'family', 'genus', 'kingdom', 'order', 'phylum'])

features = ['class', 'family', 'genus', 'kingdom', 'order', 'phylum']

classt = scaled_features['class'].unique().tolist()
family = scaled_features['family'].unique().tolist()
genus = scaled_features['genus'].unique().tolist()
kingdom = scaled_features['kingdom'].unique().tolist()
order = scaled_features['order'].unique().tolist()
phylum = scaled_features['phylum'].unique().tolist()

for a in kingdom:
    for b in phylum:
        for c in classt:
            for d in order:
                for e in family:
                        current = ordered_features['id'].loc[(ordered_features['kingdom'] == a) & (ordered_features['phylum'] == b) & (ordered_features['class'] == c) & (ordered_features['order'] == d) & (ordered_features['family'] == e)]
                        if len(current) != 0:
                            print(len(current))
                            current.to_csv(r'/Users/jas10022/Documents/GitHub/iNaturalist/' + str(a) + '_' + str(b) + '_' + str(c) + '_' + str(d) + '_' + str(e) + '.csv', index=False)

for number in numbers:
    data = lower_train_data.loc[lower_train_data['CatagoryID']==number]

    data.to_csv(r'/Users/jas10022/Documents/GitHub/iNaturalist/' + str(number) + '.csv', index=False)


#this is how we can create a data file with all the images in a 1,75,75 shape 
#modify the for loop in order to use it and the allImages are all the images file names
#update the .open method to the directory of the train images then the + im
# the Images variable will contain all the picures each resized to 75 by 75

upperNN = pd.read_csv('upperNN .csv')

file_names = upperNN['File_Name']
file_names = file_names[4999:265213]

Images = np.empty([1,75, 75])
i = 0
a = 0
for im in file_names:
        if i >= 5000:
            Images = Images[1:]
            output = open(str(a)+'.pkl', 'wb')
            pickle.dump(Images, output)
            output.close()            
            Images = np.empty([1,75, 75])
            a += 1
            i = 0
        img = Image.open(im, 'r').convert('LA')
        cover = resizeimage.resize_cover(img, [75, 75], validate=False)
        np_im = np.array(cover)
    
        pix_val_flat = np.array([x for sets in np_im for x in sets])
        train_data = pix_val_flat[:,0].astype('float64') /255
        train_data = np.resize(train_data, (1,75,75))
        
        Images = np.concatenate((Images,train_data))
        i += 1
        print(i)
        
Images = np.empty([1,75, 75])
i = 0
a = 54

for num in range(59,65):
    if i == 5:
        Images = Images[1:]
        output = open(str(a)+'.pkl', 'wb')
        pickle.dump(Images, output)
        output.close()            
        Images = np.empty([1,75, 75])
        i = 0
        a += 1
    pkl_file = open(str(num) + '.pkl', 'rb')

    data1 = pickle.load(pkl_file)

    Images = np.concatenate((Images,data1))
    
    pkl_file.close()
    print(i)
    i += 1
    
#how to read 3d array

pkl_file = open('54.pkl', 'rb')
pkl_file1 = open('55.pkl', 'rb')
pkl_file2 = open('56.pkl', 'rb')
pkl_file3 = open('57.pkl', 'rb')
pkl_file4 = open('58.pkl', 'rb')
pkl_file5 = open('58.pkl', 'rb')
pkl_file6 = open('60.pkl', 'rb')
pkl_file7 = open('61.pkl', 'rb')
pkl_file8 = open('62.pkl', 'rb')
pkl_file9 = open('63.pkl', 'rb')
pkl_file10 = open('64.pkl', 'rb')

data1 = pickle.load(pkl_file)
data2 = pickle.load(pkl_file1)
data3 = pickle.load(pkl_file2)
data4 = pickle.load(pkl_file3)
data5 = pickle.load(pkl_file4)
data6 = pickle.load(pkl_file5)
data7 = pickle.load(pkl_file6)
data8 = pickle.load(pkl_file7)
data9 = pickle.load(pkl_file8)
data10 = pickle.load(pkl_file9)
data11 = pickle.load(pkl_file10)

pkl_file.close()
    
Images = np.empty([1,75, 75])
i = 0
for num in range(0,1010):
    current = pd.read_csv('/Users/jas10022/Documents/GitHub/iNaturalist/Data/Lower_NN_Data/' + str(num) + '.csv')
    for id in current['ImageId']:
        file_num = int(id / 25000)
        index = id - (file_num * 25000)
        if file_num == 0:
            im = data1[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 1:
            im = data2[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 2:
            im = data3[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 3:
            im = data4[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 4:
            im = data5[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 5:
            im = data6[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 6:
            im = data7[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 7:
            im = data8[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 8:
            im = data9[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 9:
            im = data10[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
        if file_num == 10:
            im = data11[index].reshape(1,75,75)
            Images = np.concatenate((Images,im))
    Images = Images[1:]
    output = open(str(i) + '.pkl', 'wb')
    pickle.dump(Images, output)
    output.close()
    Images = np.empty([1,75, 75])
    print(i)
    i += 1

#to look at pictures
import matplotlib.pyplot as plt

plt.imshow(Images[2], cmap="gray")
plt.subplots_adjust(wspace=0.5)
plt.show()

#how to save 3d array
output = open('data.pkl', 'wb')
pickle.dump(Images, output)
output.close()

print("fuck yeah")