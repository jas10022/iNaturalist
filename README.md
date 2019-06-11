# iNaturalist Competition 2019

The idea behind this CNN HyperNetwork is to take the data classification given in this competition for each species and create a set of "Upper Neural Networks". There will be 6 upper networks, each of which will predict the  pictures class, family, genus, kingdom, order, phylum. Using this classification, there will then be a set of "Lower Neural Networks" where the predictions of the upper networks will feed into the group predicted by the network in order to specify the species in the group. Essencialy this method is taking a picture and making the classification simplier for the entire network to get a better accuracy.

This is a design layout of the HyperNetwork:
![Hyper Neural Network Design](https://scontent-sjc3-1.xx.fbcdn.net/v/t1.15752-9/61281631_395800384479998_6138768229000019968_n.png?_nc_cat=101&_nc_oc=AQnU4YB-zVRHBTVfZDc7UthRTfCH9P8Tu0H5wkt1vVR-Dl_PIuICVYepBdmcPuCF1SwlU1zQrrWMqyJK23FNzuIu&_nc_ht=scontent-sjc3-1.xx&oh=56f83dd4005f141c121f4520a6726421&oe=5D9D316F)

This is how the HyperNetwork will traing on all these diffrent models:
![Theory of HyperNetwork](https://scontent-sjc3-1.xx.fbcdn.net/v/t1.15752-9/60771839_643704266054313_6017675504444768256_n.png?_nc_cat=104&_nc_oc=AQl_3C0S_UcB_OisLMNymFV137wd65OMNsaQJgUe-S5E1Lz7UQFWhT1TyvQGSc7cIwayDfpTOkh8bGsaP2HCTEAx&_nc_ht=scontent-sjc3-1.xx&oh=ff9a5e5826b217a01dbfcc4ed5dba12c&oe=5D820A61)

Competition Link: https://www.kaggle.com/c/inaturalist-2019-fgvc6
