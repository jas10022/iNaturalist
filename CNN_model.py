#!/usr/bin/env pyt
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

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
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)            
              # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)            
            
              # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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
    cnn_classifier.train(input_fn=train_input_fn, steps=2000, hooks=[logging_hook])

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
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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

        

#just a start on how to automate creating all the models for the upper NN

#creating a basic model to start the model training
# this code chunk will auto run and create a model trained on all the data for the species classification
# this is all automated and may take a long time to create al 5 models as it has to go through
#25000 images 11 times as the data is split up inot 11 files for each model
# bellow is the code for Upper NN to give to ur dad to run
col_names = ['Class', 'Family','Kingdom','Order','Phylum']
#, 'Family','Kingdom','Order','Phylum'
a = 0
b = 1
output_nodes = 0
for cat in col_names:
    
    pkl_file = open('Data/UpperNN_data/64.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    
    upperNN = pd.read_csv('Data/upperNN.csv')
    Labels = upperNN[cat][249999:265214]
    output_nodes = upperNN[cat].unique().size
    
    
    #this splits the data into training and val data for the model and also reshapes the label data
    X_train, X_val, y_train, y_val = train_test_split(data1, Labels, test_size = 0.05, random_state = 0)
    y_train = np.asarray(y_train).reshape((-1,1))
    y_val = np.asarray(y_val).reshape((-1,1))
    
    model_location = CNN(X_train,y_train,X_val,y_val,output_nodes, str(cat), "upper")

    ##need prediction values

    for i in range(54,64):
        
        pkl_file = open('Data/UpperNN_data/' + str(i) + '.pkl', 'rb')
        data1 = pickle.load(pkl_file)
    
        upperNN = pd.read_csv('Data/upperNN.csv')
        Labels = upperNN[cat][a * 25000:b * 25000]
        a += 1
        b += 1
    
        #this splits the data into training and val data for the model and also reshapes the label data
        X_train, X_val, y_train, y_val = train_test_split(data1, Labels, test_size = 0.05, random_state = 0)
        y_train = np.asarray(y_train).reshape((-1,1))
        y_val = np.asarray(y_val).reshape((-1,1))
                    
        #this session will open up any saved model created in directory and will run prediction on that
        # you can also train with it using the training lines
        with tf.Session() as sess:
          # Restore variables from disk.
            currentCheckpoint = tf.train.latest_checkpoint(cat)
            saver = tf.train.import_meta_graph(currentCheckpoint + ".meta")
            saver.restore(sess, currentCheckpoint)
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
        
                #change steps to 20000
            sunspot_classifier.train(input_fn=train_input_fn, steps=2000)
            
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
    
#this will be the lower NN code to run
#this code is very basic where it will look into the Data/Sorted_species files one by one as these
# files contains the groups for all the species and it will train one model for each of the groupings
# once read it will look at all the species ids and then look into the Data/Train_data file to get all 
# the picture data and store itinto train_data for each species
# the code will also take the labels from the Data/Lower_NN_Data files to create the Labels for the training
# then after the code splits the data created into training and validation data to train the model
output_nodes_number = 0

for filename in os.listdir("Data/Sorted_species"):
    if filename.endswith(".csv"):
         data = pd.read_csv("Data/Sorted_species/" + filename, names=['a'])
         output_nodes_number = data.size 
         name = os.path.splitext(filename)[0]
         train_data = np.empty([1,75, 75])
         Labels = np.empty([1])
         for species in data['a']:
                
             pkl_file = open("Data/Train_data/" + str(species) + '.pkl', 'rb')
             data1 = pickle.load(pkl_file)
             train_data = np.concatenate((train_data,data1))
             pkl_file.close()
             
             current_labels = pd.read_csv("Data/Lower_NN_Data/" + str(species) + '.csv')["CatagoryID"]
             Labels = np.concatenate((Labels,current_labels.values))
        
         Labels = Labels[1:]
         train_data = train_data[1:]
         
         labelencoder = LabelEncoder()
         Labels[0:] = labelencoder.fit_transform(Labels[0:])

         X_train, X_val, y_train, y_val = train_test_split(train_data, Labels, test_size = 0.20, random_state = 0)
         y_train = np.asarray(y_train).astype('int32').reshape((-1,1))
         y_val = np.asarray(y_val).astype('int32').reshape((-1,1))
        
         model_location = CNN(X_train,y_train,X_val,y_val,output_nodes_number, name, "lower")


def cnn_model_lower(features, labels, mode):
            
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
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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
