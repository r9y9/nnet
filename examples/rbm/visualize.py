# coding: utf-8
import json

# A weight visualization script for Restricted Bolztmann Machines
# trained on MNSIT dataset.

import numpy as np
import sys
from pylab import *

argv = sys.argv
if len(argv) != 2:
    print "Incorrect arguments."
    quit()
 
filename = argv[1]
f = open(filename)
data = json.load(f)
f.close()
 
W = np.array(data['W'])
M = int(data['NumHiddenUnits'])
w,h = int(np.sqrt(M)), int(np.sqrt(M))
M = w*h
for i in range(M):
    subplot(w, h, i+1)
    imshow(W[i].reshape(28, 28), cmap=cm.Greys_r)
    xticks(())
    yticks(())
    
show()
