from __future__ import division
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from nltk.corpus import stopwords
import csv
#from sklearn import tree
import pdb
import string

ps = PorterStemmer()

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

def maxsensefreq(synsets):
    max= 0
    for s in synsets:
        freq = 0  
        for lemma in s.lemmas():
            freq += lemma.count()
        if freq > max:
            max = freq
    return max    

def diffmaxsensefreq(synsets):
    max = 0
    secondmax = 0
    for s in synsets:
        freq = 0
        for lemma in s.lemmas():
            freq += lemma.count()
        if freq > max:
            max = freq
        elif freq > secondmax:
            secondmax = freq
    return max-secondmax


def callLesk(word_array):

    wordtrain = []
    label = []
    sent = word_array
    trainset = {}
    for idx, ew in enumerate(sent):
        answer = lesk(sent,ew)
        l = []
        ss = wn.synsets(ew)
        
        c = len(ss)
        l.append(c)
        l.append(abs(int(answer[3])-int(answer[1])))
        l.append(maxsensefreq(ss))
        l.append(diffmaxsensefreq(ss))
        #l.append(int(answer[3]))
        #trainset.append(l)
        #label.append(0)
        trainset[idx] = l
    return trainset

def combinedict(dict1,dict2):
    dict = {}
    if len(dict1) != len(dict2):
        print "Length not same"
        return dict1
    for idx,val in enumerate(dict1):
        l = dict1[idx]
        l.extend(dict2[idx])
        #pdb.set_trace()
        dict[idx] = l
    return dict

t1 = callLesk(['I','have','an','apple'])
t2 = callLesk(['I','have','a','pen'])
t = combinedict(t1,t2)
#print t1
#print t2
#print t
