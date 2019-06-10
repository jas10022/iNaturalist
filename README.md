# iNaturalist Competition 2019

#### The idea behind this CNN HyperNetwork is to take the data classification given in this competition for each species and create a set of "Upper Neural Networks". There will be 6 upper netwokrs, each of which will predict the  pictures class, family, genus, kingdom, order, phylum. Using this classification, there will then be a set of "Lower Neural Networks" where the predictions of the upper networks will feed into the group predicted by the network in order to specify the species in the group. Essencialy this method is taking a picture and making the classification simplier for the entire network to get a better accuracy.


Competition Link: https://www.kaggle.com/c/inaturalist-2019-fgvc6
