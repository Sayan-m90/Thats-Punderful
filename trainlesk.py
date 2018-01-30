from __future__ import division
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from nltk.corpus import stopwords
import csv
from sklearn import tree
import pdb
import string
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#bank_sents = [
#'The batteries were given out free of charge',
#'I went to the bank to deposit my money',
#'The river bank was full of dead fishes']

#plant_sents = ['A few boxers were standing in a line. That\'s the punch line!',
#'A bicycle cannot stand alone; it is two tired.',
#'The workers at the industrial plant were overworked',
#'The plant had no flowers or leaves']

ps = PorterStemmer()
#reader = csv.reader(f, delimiter=" ")


# text_file = open("training_set.ods", "r")
# list1 = text_file.readlines()
# listtext = []
# for item in list1:
# 	listtext.append(str(item))
# 	print item

def lesk(context_sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
    max_overlaps = 0; lesk_sense = None
    sec_max_overlaps = 0; lesk_sense_2 = None
    thi_max_overlaps = 0; lesk_sense_3 = None
    #context_sentence = context_sentence.split()
    lesk = []
    for ss in wn.synsets(ambiguous_word):
        # If POS is specified.
        if pos and ss.pos() is not pos:
			print ss.pos
			print pos
			print "continuing"
			continue

        lesk_dictionary = []

        # Includes definition.
        lesk_dictionary+= ss.definition().split()
        # Includes lemma_names.
        lesk_dictionary+= ss.lemma_names()

        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))       

        if stem == True: # Matching exact words causes sparsity, so lets match stems.
            lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
            context_sentence = [ps.stem(i) for i in context_sentence] 

    	overlaps = set(lesk_dictionary).intersection(context_sentence)
        if len(overlaps) >  max_overlaps:
			lesk_sense = ss
			max_overlaps = len(overlaps)
        elif len(overlaps) > sec_max_overlaps:
            lesk_sense_2  = ss
            sec_max_overlaps = len(overlaps)
        elif len(overlaps) > thi_max_overlaps:
        	thi_max_overlaps = len(overlaps)
        	lesk_sense_3  = ss
    lesk.append(lesk_sense)
    lesk.append(str(max_overlaps))
    lesk.append(lesk_sense_2)
    lesk.append(str(sec_max_overlaps))
    lesk.append(lesk_sense_3)
    lesk.append(str(thi_max_overlaps))

    return lesk

wordtrain = []
label = []
#pdb.set_trace()
with open('training_set.tsv') as f:
    reader = csv.reader(f, delimiter="\t")
    for lidx, line in enumerate(reader):
        #print lidx
        sent = line[0]
        sent = sent.translate(None, string.punctuation)
        if len(line) == 2:
            print "Quit on line: {0}".format(lidx)
            break
        punidx =  line[len(line)-1] 
        punidxs = punidx.split(",")
        if len(punidxs) is 0:
            print "Quit on line: {0}".format(lidx)

        punidxs = [int(x) for x in punidxs]
        newpunidxs = []
        for i in punidxs:
            if i < 0:
                newpunidxs += [len(sent.split(" "))+i]
            else:
                newpunidxs += [i]
        punidxs = newpunidxs

        #test if started with any negatives
        #x = [1 for i in punidxs if int(i) < 0]
        #if 1 in x:
        #    pdb.set_trace()
        trainset = []
        for idx, ew in enumerate(sent.split()):
            if int(idx) in punidxs:
                answer = lesk(sent,ew)
                c = len(wn.synsets(ew))
                l = []
                l.append(c)
                l.append(abs(int(answer[3])-int(answer[1])))
                trainset.append(l)
                label.append(1)
            else:
                answer = lesk(sent,ew)
                l = []
                c = len(wn.synsets(ew))
                l.append(c)
                l.append(abs(int(answer[3])-int(answer[1])))
                #l.append(int(answer[3]))
                trainset.append(l)
                label.append(0)
        wordtrain.append(trainset)
for i in wordtrain:
    print i
    print "\n"
#svc = SVC(C=1.0)
#svc = tree.DecisionTreeClassifier()
#params = [{'kernel':['linear','rbf'],'random_state':[1,2,5,10],'gamma':[1e0, 1e-1, 1e-3, 1e-5],'C':[1,5,10,100]}]
#clf = GridSearchCV(svc,params,cv=5) 
#clf = tree.DecisionTreeClassif)
#svc.fit(trainset, label)
#trainpreds = svc.predict(trainset)
#cm = confusion_matrix(trainpreds, label)
#print(cm)
#mzip = zip(trainpreds,label)
#corrects = [1 for x in mzip if x[0]==x[1]]
#accuracy = sum(corrects) / len(label)
#print accuracy
#print trainpreds
#clf.predict_proba([[2, 2]])
#y_pred = clf.predict([[2,2]])
#print(y_pred)

