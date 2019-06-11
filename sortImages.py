
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

# get a list of names
numbers=lower_train_data['CatagoryID'].unique().tolist()

for number in numbers:
    data = lower_train_data.loc[lower_train_data['CatagoryID']==number]

    data.to_csv(r'/Users/jas10022/Documents/GitHub/iNaturalist/' + str(number) + '.csv', index=False)


#this is how we can create a data file with all the images in a 1,75,75 shape 
#modify the for loop in order to use it and the allImages are all the images file names
#update the .open method to the directory of the train images then the + im
# the Images variable will contain all the picures each resized to 75 by 75

upperNN = pd.read_csv('upperNN .csv')

file_names = upperNN['File_Name']

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
        
Images = Images[1:]
output = open(str(a)+'.pkl', 'wb')
pickle.dump(Images, output)
output.close()
        
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

# example of how to implement
# make sure when we group all the training data that the data is seperated into 
# training and test data to make sure the model is accurate on a new set of data that
# the neural network has not see yet

#this data is already in 28 x 28 format so it did not need to resize the pictures
#also this data is also gray scaled already so we dont need to apply it to this either
#retrieving the data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

test_data = pd.read_csv("test.csv")

test_data = test_data.values.reshape((-1, 28, 28, 1))
test_data = test_data.astype('float32') /255

#this line creates and trains the entire model the inputs are below and must be given 
# an output number of nodes, for this there is 10
model_location = CNN(train_data,train_labels,eval_data,eval_labels,10, "save", "lower")


#Bellow is how to train the UpperNN model for the catagories, I gave the example for the Kingdom

#this reads the first file of image data
#each of the files have 25000 images except for the last one
pkl_file = open('54.pkl', 'rb')
data1 = pickle.load(pkl_file)
pkl_file.close()

#This reads the Label data and extracts the specific values from the file read above
upperNN = pd.read_csv('Data/upperNN.csv')
Labels = upperNN['Kingdom'][0:25000]


#this will run and create the model in the local working directory
model_location = CNN(X_train,y_train,X_val,y_val,1, "Kingdom_Model", "upper")

cat = "Kingdom_Model"
with tf.Session() as sess:
          # Restore variables from disk.
    saver = tf.train.import_meta_graph(cat + "/model.ckpt-" + str(b * 10) + ".meta")
    saver.restore(sess, cat + "/model.ckpt-" + str(b * 10))
    print("Model restored.")
          
    sunspot_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_test, model_dir=cat)
        
        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
          
          ## train here
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                   x={"x": X_train},
                   y=y_train,
                   batch_size=100,
                   num_epochs=None,
                   shuffle=True)

        # Train one step and display the probabilties
    sunspot_classifier.train(
            input_fn=train_input_fn,
            steps=1,
            hooks=[logging_hook])
        #change steps to 20000
    sunspot_classifier.train(input_fn=train_input_fn, steps=10)
    
        # Evaluation of the neural network
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_val},
            y=y_val,
            num_epochs=1,
            shuffle=False)
    
    eval_results = sunspot_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


print("fuck yeah")

 # predict with the model and print results
 pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_val},
    shuffle=False)
    pred_results = sunspot_classifier.predict(input_fn=pred_input_fn)
     
    pred_class = np.array([p['class_ids'] for p in pred_results]).squeeze()
    print(pred_class)