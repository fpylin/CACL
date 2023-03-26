#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import pandas as pd
import pickle


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
parser.add_option("-P", "--penalty", dest="penalty", default='l2',
                  help="Penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’} [default: %default]", metavar="NAME")
parser.add_option("-C", "--C", dest="C", default=1.0,
                  help="C [default: %default]", metavar="NAME")
parser.add_option("-L", "--l1-ratio", dest="l1_ratio", default=None,
                  help="L1 ratio (between 0 and 1) [default: %default]", metavar="NAME")
parser.add_option("-S", "--solver", dest="solver", default='lbfgs',
                  help="Solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} [default: %default]", metavar="NAME")
parser.add_option("-I", "--max-iter", dest="max_iter", default=50000,
                  help="Maximum number of iterations taken for the solvers to converge. [default: %default]", metavar="NAME")

(options, args) = parser.parse_args()

if len(args) != 3:
    print(options);
    print(args);
    parser.error("Wrong number of arguments")
    sys.exit(1);
 
action   = args[0]
fn_model = args[1]
fn_data  = args[2]

class_name = options.class_name

data_x = pd.read_table(fn_data, header=0)
data_y = data_x.pop(class_name)

if options.l1_ratio != None:
    options.l1_ratio = float(options.l1_ratio)

# eprint(data_x)

# eprint("Class name:  " + class_name)
# eprint("Solver:      " + str(options.solver) )
# eprint("Penalty:     " + str(options.penalty) )
# eprint("C:           " + str(options.C) )
# eprint("L1 ratio:    " + str(options.l1_ratio) )
# eprint("Max iter:    " + str(options.max_iter) )

#eprint(data_x)

if action == 'train':
#    nc = data_y.unique().count()  
 #   labels = np.zeros( (data_y.shape[0], nc) )
  #  for r in range(data_y.shape[0]):
   #     z = data_y[r]
    #    #print(r,z);
     #   labels[r, z] = 1
      #  #print(labels)

        
    imputor = SimpleImputer(strategy="most_frequent")
    imputor.fit(data_x)
    data_x = imputor.transform(data_x)

    model = LogisticRegression(penalty=options.penalty, C=float(options.C), l1_ratio=options.l1_ratio, solver=options.solver, max_iter=int(options.max_iter) )

    # eprint(model)
    
    score = model.fit(data_x, data_y)
    
    eprint( "Accuracy: " + str( model.score(data_x, data_y) ) )
    
    with open(fn_model, 'wb') as file:
        pickle.dump( (imputor, model), file)
    
elif action == 'predict':
    with open(fn_model, 'rb') as file:
        imputor, model = pickle.load(file)	

    data_x = imputor.transform(data_x)
    predictions = model.predict_proba(data_x)
    
    nc = predictions.shape[1]
    
    i=0
    for pred_dict in predictions:
        p = list( map(lambda c: pred_dict[c], range(nc)) )
        print(*p, sep="\t")
        i += 1
