#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import pandas as pd
import pickle

from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, ComplementNB
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier

import random

# import logging

# logging.getLogger().setLevel(logging.INFO)
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from optparse import OptionParser

parser = OptionParser()
parser.usage = "usage: %prog [options] train|predict filename.model data_file.tsv"
parser.add_option("-y", "--class-name", dest="class_name", default="__class__",
                  help="Class name [default: %default]", metavar="NAME")
parser.add_option("-T", "--type", dest="clftype", default="CategoricalNB",
                  help="CategoricalNB, GaussianNB, BernoulliNB, ComplementNB [default: %default]", metavar="NAME")

(options, args) = parser.parse_args()

if len(args) != 3:
    parser.error("Wrong number of arguments")
    sys.exit(1);
 
action   = args[0]
fn_model = args[1]
fn_data  = args[2]

class_name = options.class_name

data_x = pd.read_table(fn_data, header=0)

if len(data_x.columns) == 1: # adding this to prevent creating an empty matrix
	data_y = data_x
else:
	data_y = data_x.pop(class_name)



if action == 'train':
    #    nc = data_y.unique().count()  
    #   labels = np.zeros( (data_y.shape[0], nc) )
    #  for r in range(data_y.shape[0]):
    #     z = data_y[r]
    #    #print(r,z);
    #   labels[r, z] = 1
    #  #print(labels)

    if len(data_x.columns) == 0:
        dummy_clf = DummyClassifier(strategy='prior')
        model = dummy_clf.fit(data_x, data_y)
        imputor = None
    else:
        imputor = SimpleImputer(strategy="most_frequent")
        imputor.fit(data_x)
        data_x = imputor.transform(data_x)

        if options.clftype == "CategoricalNB":
            model = CategoricalNB()
        elif options.clftype == "GaussianNB":
            model = GaussianNB()
        elif options.clftype == "ComplementNB":
            model = ComplementNB()
        elif options.clftype == "BernoulliNB":
            model = BernoulliNB()
            
        # eprint(data_y)
        score = model.fit(data_x, data_y)
        #  eprint(model)
    
    eprint( str(model) + " -> " + class_name + ". Accuracy: " + str( model.score(data_x, data_y) )  )
        
    with open(fn_model, 'wb') as file:
        pickle.dump( (imputor, model), file)
    
elif action == 'predict':
	
    with open(fn_model, 'rb') as file:
        imputor, model = pickle.load(file)	

    if imputor != None:
        data_x = imputor.transform(data_x)
    
    eprint( "Loaded model " + str(model) + ". " + str( data_x.shape ) ) 

    predictions = model.predict_proba(data_x)
    
    nc = predictions.shape[1]
    
    i=0
    for pred_dict in predictions:
        p = list( map(lambda c: pred_dict[c], range(nc)) )
        print(*p, sep="\t")
        i += 1
