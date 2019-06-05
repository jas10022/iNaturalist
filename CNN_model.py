#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:21:22 2019

@author: jas10022
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from skimage import transform 
from skimage.color import rgb2gray
import pandas as pd
from PIL import Image
from resizeimage import resizeimage
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

def CNN(train_data, train_labels, eval_data, eval_labels, output_nodes_number, model_name, model_type):
      
    if model_type == "lower":
        #tensorflow model function for lowerNN
        def cnn_model(features, labels, mode):
            
            input_layer = tf.reshape(features["x"], [-1, 75, 75,1])
            
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                  inputs=input_layer,
                  filters=32,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            
              # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            
              # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                  inputs=pool1,
                  filters=64,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            
              # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1,  18 * 18 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            
              # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=output_nodes_number)
            
            predicted_classes =tf.argmax(input=logits, axis=1)
            predictions = {
                        'class_ids': predicted_classes[:, tf.newaxis],
                        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
                        'logits': logits,
                    }
            export_outputs = {
              'prediction': tf.estimator.export.PredictOutput(predictions)
              }
            if mode == tf.estimator.ModeKeys.PREDICT:  
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
            
              # Calculate Loss (for both TRAIN and EVAL modes)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits = logits)
            
              # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            
              # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                  "accuracy": tf.metrics.accuracy(
                      labels=labels, predictions=predictions["class_ids"])
            }
            return tf.estimator.EstimatorSpec(
                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    if model_type == "upper":
        #tensorflow model function for upperNN
        def cnn_model(features, labels, mode):
            
            input_layer = tf.reshape(features["x"], [-1, 75, 75, 1])
            
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                  inputs=input_layer,
                  filters=32,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            
              # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            
              # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                  inputs=pool1,
                  filters=64,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            
              # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 18 * 18 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            
              # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=output_nodes_number)
            
            predicted_classes =tf.argmax(input=logits, axis=1)
            predictions = {
                        'class_ids': predicted_classes[:, tf.newaxis],
                        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
                        'logits': logits,
                    }
            export_outputs = {
              'prediction': tf.estimator.export.PredictOutput(predictions)
              }
            if mode == tf.estimator.ModeKeys.PREDICT:  
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
            
              # Calculate Loss (for both TRAIN and EVAL modes)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits = logits)
            
              # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            
              # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                  "accuracy": tf.metrics.accuracy(
                      labels=labels, predictions=predictions["class_ids"])
            }
            return tf.estimator.EstimatorSpec(
                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    #This is where we need to load up the data for each group

    ModelDir = model_name
    # Create the Estimator
    
    run_config = tf.contrib.learn.RunConfig(
    model_dir=ModelDir,
    keep_checkpoint_max=1)
    
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model, config=run_config)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    #change steps to 20000
    cnn_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    # Evaluation of the neural network
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
        #instead of predicting on a test data set we will save the model
    #model_dir = cnn_classifier.export_savedmodel(
       # model_name,
        #serving_input_receiver_fn=serving_input_receiver_fn)

    return  ModelDir
            
def cnn_model_test(features, labels, mode):
            
            input_layer = tf.reshape(features["x"], [-1, 75, 75,1])
            
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                  inputs=input_layer,
                  filters=32,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            
              # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            
              # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                  inputs=pool1,
                  filters=64,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            
              # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1,  18 * 18 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            
              # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=1)
            
            predicted_classes =tf.argmax(input=logits, axis=1)
            predictions = {
                        'class_ids': predicted_classes[:, tf.newaxis],
                        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
                        'logits': logits,
                    }
            export_outputs = {
              'prediction': tf.estimator.export.PredictOutput(predictions)
              }
            if mode == tf.estimator.ModeKeys.PREDICT:  
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
            
              # Calculate Loss (for both TRAIN and EVAL modes)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits = logits)
            
              # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            
              # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                  "accuracy": tf.metrics.accuracy(
                      labels=labels, predictions=predictions["class_ids"])
            }
            return tf.estimator.EstimatorSpec(
                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    


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

 # predict with the model and print results
 pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_val},
    shuffle=False)
    pred_results = sunspot_classifier.predict(input_fn=pred_input_fn)
     
    pred_class = np.array([p['class_ids'] for p in pred_results]).squeeze()
    print(pred_class)

        

#just a start on how to automate creating all the models for the upper NN

#creating a basic model to start the model training
# this code chunk will auto run and create a model trained on all the data for the species classification
# this is all automated and may take a long time to create al 5 models as it has to go through
#25000 images 11 times as the data is split up inot 11 files for each model
# bellow is the code for Upper NN to give to ur dad to run
col_names = ['class', 'family','kingdom','order','phylum']

a = 0
b = 1
for cat in col_names:
    
    pkl_file = open('64.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    
    upperNN = pd.read_csv('Data/upperNN.csv')
    Labels = upperNN[cat][250000:265213]
    
    
    #this splits the data into training and val data for the model and also reshapes the label data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(data1, Labels, test_size = 0.05, random_state = 0)
    y_train = np.asarray(y_train).reshape((-1,1))
    y_val = np.asarray(y_val).reshape((-1,1))
    
    model_location = CNN(X_train,y_train,X_val,y_val,1, str(cat), "upper")

    ##need prediction values

    for i in range(54,64):
        
        pkl_file = open(str(i) + '.pkl', 'rb')
        data1 = pickle.load(pkl_file)
    
        upperNN = pd.read_csv('Data/upperNN.csv')
        Labels = upperNN[cat][a * 25000:b * 25000]
        a += 1
        b += 1
    
        #this splits the data into training and val data for the model and also reshapes the label data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(data1, Labels, test_size = 0.05, random_state = 0)
        y_train = np.asarray(y_train).reshape((-1,1))
        y_val = np.asarray(y_val).reshape((-1,1))
                    
        #this session will open up any saved model created in directory and will run prediction on that
        # you can also train with it using the training lines
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
    if b == 11:
        break
   