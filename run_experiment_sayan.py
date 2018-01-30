
from leskfeature import *
from evan_feats import *
from sklearn import svm, preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle
import csv
import pdb
import time

try:
    ppdbtoks = pickle.load(open("ppdbtoks.p",'rb'))
except IOError:
    print "ppdb preprocessing..."
    ppdbtoks = ppdb_pretrain()
    pickle.dump(ppdbtoks, open("ppdbtoks.p",'wb'))
print "got ppdb preprocessing info"

try:
    bc = pickle.load(open("bc.p",'rb'))
    bwc = pickle.load(open("bwc.p",'rb'))
    tc = pickle.load(open("tc.p",'rb'))
    twc = pickle.load(open("twc.p",'rb'))
except IOError:
    print "preprocessing pmi collocations..."
    bc, bwc, tc, twc = pmi_colloc_pretrain()
    pickle.dump(bc,open("bc.p",'wb'))
    pickle.dump(bwc,open("bwc.p",'wb'))
    pickle.dump(tc,open("tc.p",'wb'))
    pickle.dump(twc,open("twc.p",'wb'))
print "got pmi preprocessing info"

print "Beginning feature calculation..."

trainfeats = [] #list of word-feature lists
labels = [] #by word
with open('training_set.tsv','r') as f:
    for line in csv.reader(f, delimiter="\t"):
        sent = line[0]
        #print sent
        #if len(line) < 3:
        #    break
        
        #wordarray = "I have a dog.".split(" ")
        wordarray = sent.split(" ")
        #read labels

        lfeats = callLesk(wordarray)
        stopfeat = isstop(wordarray)
        polyfeats = polysemy_feats(wordarray)
        #posfeat = relsentpos(wordarray)
        ppdbfeats = ppdb_feats(ppdbtoks, wordarray)
        pmifeats = pmi_feats(bc, bwc, tc, twc, wordarray)


        allfeats = combinedict(lfeats,stopfeat)
        allfeats = combinedict(allfeats,polyfeats)
        #allfeats = combinedict(allfeats,posfeat)
        allfeats = combinedict(allfeats,ppdbfeats)
        allfeats = combinedict(allfeats,pmifeats)
        
        for idx in range(len(wordarray)):
            #add feats for each word
            trainfeats.append(allfeats[idx])

        #add label for each word
        if len(line) < 3:
            continue
        punidxs = line[2].split(",")
        punidxs = [int(x) for x in punidxs]
        newpunidxs = []

        for i in punidxs:
            if i<0:
                newpunidxs += [len(sent.split(" "))+i]
            else:   newpunidxs += [i]
        punidxs = newpunidxs

        if idx in punidxs:
            labels.append(1)
        else:
            labels.append(0)

#print trainfeats[:10]
#print labels[:10]

trainfeats = np.array(trainfeats)
X = trainfeats
X = preprocessing.scale(X)
print "finished generating features"

#X = normalize(trainfeats,axis=0) #normalize features across all samples
#scaler = preprocessing.StandardScaler().fit(trainfeats)
#X = scaler.transform(trainfeats)

#SVM

for kval in ['rbf']:
    for cval in [1]:
        start = time.time()
        clfSVM = svm.SVC(probability=True, random_state=0, kernel=kval, C=cval)
        
        clfSVM.fit(X[0:len(labels)][:],labels)
        pred = clfSVM.predict(X)
        #cvscoresSVM = cross_val_score(clfSVM, X, labels, scoring='f1')
        #print "SVM k:{0}, C:{1}, crossvalscore: {2}".format(kval,cval,cvscoresSVM)
        end = time.time()
        #print np.sum(1)
        count1 = 0
        count2 = 0
        for i,s in enumerate(pred):
            if int(s) == 1:
                #print i
                count1 = count1 +1
                if int(i) < 1847:
                    count2 = count2 + 1
        print "svm time: {0}".format(end - start)
        print "count1:{0}".format(count1)
        print "count2:{0}".format(count2)

#Decision Trees
maxclfDT = None
maxclDTscore = 0

for crit in ["gini","entropy"]:
    start = time.time()
    clfDT = DecisionTreeClassifier(criterion=crit) #n_estimators=nestimators, random_state=0)
    cvscoresDT = cross_val_score(clfDT, X, labels, scoring='f1')
    print "DT: criterion:{0}, crossvalscore: {1}".format(crit, cvscoresDT)
    end = time.time()
    print "dt time: {0}".format(end - start)
    score = np.mean(np.array(cvscoresDT))
    if score > maxclDTscore:
        maxclfDT = clfDT
        maxclDTscore = score

#AdaBoost Trees
ABC = AdaBoostClassifier(base_estimator=maxclfDT)
cvscoresADT = cross_val_score(ABC, X, labels, scoring='f1')
print "AdaBoostDT crossvalscore: {0}".format(cvscoresADT)
