from __future__ import division
import nltk
from nltk.collocations import *
from nltk.corpus import *
import string
import re
import pdb

def polysemy_feats(wordarray):
    efeats = {}
    #what to do for OOVs in wordnet?
    for idx,mtoken in enumerate(wordarray):
        efeats[idx] = []
        mtoken = cleanword(mtoken)
        mlen = len(nltk.corpus.wordnet.synsets(mtoken))
        if mtoken not in stopwords.words():
            #gives raw sense count
            efeats[idx].append(mlen) 

            #binary indicator of any polysemy
            if mlen > 1:
                efeats[idx].append(1) 
            else:
                efeats[idx].append(0) 

            #bin 1
            if mlen > 1 and mlen <= 10:
                efeats[idx].append(1)
            else:
                efeats[idx].append(0)

            #bin 2
            if mlen > 10 and mlen <= 20:
                efeats[idx].append(1)
            else:
                efeats[idx].append(0)

            #bin 3
            if mlen > 20:
                efeats[idx].append(1)
            else:
                efeats[idx].append(0)

        else: #stopword
            efeats[idx] = [1,0,0,0,0] #one sense, not in larger bins

    return efeats

def cleanword(word):
    #lower case, remove excess punctuation
    word = word.lower()
    word = word.strip()
    word = word.strip(string.punctuation)
    return word

def read_ppdb(ftype):
    print 'reading in ppdb filetype: {0}'.format(ftype)
    bn = "/home/scratch/jaffe/pun_detection/resources/"
    fnmap = {'so2m':bn+'ppdb-1.0-s-o2m',
             'sm2o':bn+'ppdb-1.0-s-m2o',
             'sp':bn+'ppdb-1.0-s-phrasal',
             'si':bn+'ppdb-1.0-s-phrasal-self'}
    fn = fnmap[ftype]
    mset = set()
    #pdb.set_trace()
    with open(fn, 'r') as iff:
        for line in iff:
            #get words, add to mset
            if ftype in ['sm2o', 'si', 'sp']:
                phrase = re.search(r'.*\|\|\|(.*)\|\|\|.*\|\|\| Abstract.*', line).group(1)
                words = phrase.strip().split(" ")
                for word in words:
                    mset.add(word)
            if ftype in ['so2m', 'sp']:
                phrase = re.search(r'.*\|\|\|.*\|\|\|(.*)\|\|\| Abstract.*', line).group(1)
                words = phrase.strip().split(" ")
                for word in words:
                    mset.add(word)
    return mset

def ppdb_pretrain():
    #read in ppdb files, collect words into set
    so2m = read_ppdb('so2m') #returns set of words
    sm2o = read_ppdb('sm2o')
    sp = read_ppdb('sp')
    #si = read_ppdb('si')
    ppdbtoks = so2m | sm2o | sp #| si #set union operator
    return ppdbtoks

def ppdb_feats(ppdbtoks, wordarray):
    feats = {}
    for idx,word in enumerate(wordarray):
        feats[idx] = []
        word = cleanword(word)
        if word in ppdbtoks:
            feats[idx].append(1)
        else:
            feats[idx].append(0)
    return feats

def pmi_colloc_pretrain():
    #word is in a collocation/idiom (as measured by bigram/trigram PMI using python nltk)
    #see http://www.nltk.org/howto/collocations.html
    #setup
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    #define mtrainingtoks (ideally reddit or similar domain, and HUGE)
    mtrainingtoks = [] 
    mtrainingtoks += webtext.words() #array of ~300k words
    #mtrainingtoks += movie_reviews.words() #array of ~1.5M words
    #mtrainingtoks += brown.words() # array of ~1M words
    mtrainingtoks += nps_chat.words() #array of ~45k words
    #mtrainingtoks += gutenberg.words() #array of ~2.6M words
    print "PMI training size in words: {0}".format(len(mtrainingtoks))
    #tokens = nltk.wordpunct_tokenize(text) #text is string

    print 'building bigram finder...'
    #N best bigrams by pmi
    N = 1000
    bfinder = BigramCollocationFinder.from_words(mtrainingtoks)
    bfinder.apply_freq_filter(3)
    bfinder.apply_word_filter(lambda x: len(x) == sum([1 for char in x if char in string.punctuation])) #remove all punct 'words'
    bfinder.apply_word_filter(lambda x: x.isupper()) #remove all upper-case words
    bfinder.apply_word_filter(lambda x: any(char.isdigit() for char in x)) #remove anything with digits
    bfinder.apply_word_filter(lambda x: x.startswith('http://')) #remove web addresses
    bigram_collocations = bfinder.nbest(bigram_measures.pmi, N)
    #pdb.set_trace()

    print 'building split bigram finder...'
    #N best bigram (possibly discontinuous) by pmi
    N = 1000
    bwfinder = BigramCollocationFinder.from_words(mtrainingtoks, window_size=10)
    bwfinder.apply_freq_filter(3)
    bwfinder.apply_word_filter(lambda x: len(x) == sum([1 for char in x if char in string.punctuation])) #remove all punct 'words'
    bwfinder.apply_word_filter(lambda x: x.isupper()) #remove all upper-case words
    bwfinder.apply_word_filter(lambda x: any(char.isdigit() for char in x)) #remove anything with digits
    bwfinder.apply_word_filter(lambda x: x.startswith('http://')) #remove web addresses
    bigram_split_collocations = bwfinder.nbest(bigram_measures.pmi, N)

    print 'building trigram finder...'
    #N best trigrams by pmi
    N = 1000
    tfinder = TrigramCollocationFinder.from_words(mtrainingtoks)
    tfinder.apply_freq_filter(3)
    tfinder.apply_word_filter(lambda x: len(x) == sum([1 for char in x if char in string.punctuation])) #remove all punct 'words'
    tfinder.apply_word_filter(lambda x: x.isupper()) #remove all upper-case words
    tfinder.apply_word_filter(lambda x: any(char.isdigit() for char in x)) #remove anything with digits
    tfinder.apply_word_filter(lambda x: x.startswith('http://')) #remove web addresses

    trigram_collocations = tfinder.nbest(trigram_measures.pmi, N)

    print 'building split trigram finder...'
    #N best trigram (possibly discontinuous) by pmi
    N = 1000
    twfinder = TrigramCollocationFinder.from_words(mtrainingtoks, window_size=10)
    twfinder.apply_freq_filter(3)
    twfinder.apply_word_filter(lambda x: len(x) == sum([1 for char in x if char in string.punctuation])) #remove all punct 'words'
    twfinder.apply_word_filter(lambda x: x.isupper()) #remove all upper-case words
    twfinder.apply_word_filter(lambda x: any(char.isdigit() for char in x)) #remove anything with digits
    twfinder.apply_word_filter(lambda x: x.startswith('http://')) #remove web addresses
    trigram_split_collocations = twfinder.nbest(trigram_measures.pmi, N)
    return bigram_collocations, bigram_split_collocations, trigram_collocations, trigram_split_collocations

def flatten(l):
   return [item for sublist in l for item in sublist] 

def pmi_feats(bc, bwc, tc, twc, wordarray):
    feats = {}
    for idx,wt in enumerate(wordarray):
        feats[idx] = []
        wt = cleanword(wt)
        #in bigram: binary
        if wt in flatten(bc):
                feats[idx].append(1)
        else:
                feats[idx].append(0)

        #in bigram: binary
        if wt in flatten(bwc):
                feats[idx].append(1)
        else:
                feats[idx].append(0)

        #in trigram: binary
        if wt in flatten(tc):
                feats[idx].append(1)
        else:
                feats[idx].append(0)

        #in trigram: binary
        if wt in flatten(twc):
                feats[idx].append(1)
        else:
                feats[idx].append(0)
    return feats
    '''
    #a in bigram a,b: binary
    if [wt, wtplus1] in bigram_collocations:
            feats['ainbi'] = True
        else:
            feats['ainbi'] = False


    #b in bigram a,b: binary
    if [wtminus1, wt] in bigram_collocations:
            feats['binbi'] = True
        else:
            feats['binbi'] = False

    '''

    '''
    #a in trigram abc: binary
    if [wt, wtplus1, wtplus2] in trigram_collocations:
            feats['aintri'] = True
        else:
            feats['aintri'] = False


    #b in trigram abc: binary
    if [wtminus1, wt, wtplus1] in trigram_collocations:
            feats['bintri'] = True
        else:
            feats['bintri'] = False


    #c in trigram abc: binary
    if [wtminus2, wtminus1, wt] in trigram_collocations:
        feats['cintri'] = True
    else:
        feats['cintri'] = False 
    '''
def relsentpos(wordarray):
    feats = {}
    for idx in range(len(wordarray)):
        feats[idx] = [idx/(len(wordarray)-1)]
    for idx in feats:
        if feats[idx][0] <= 0.33:
            feats[idx].append(1) #first pos is true
        else:
            feats[idx].append(0) #first pos is false

        if feats[idx][0] <= 0.66:
            feats[idx].append(1) #second pos is true
        else:
            feats[idx].append(0) #second pos is false

        if feats[idx][0] > 0.66:
            feats[idx].append(1) #third pos is true
        else:
            feats[idx].append(0) #third pos is false
    return feats

def isstop(wordarray):
    feats = {}
    for idx, word in enumerate(wordarray):
        word = cleanword(word)
        if word in stopwords.words():
            feats[idx] = [1] #is stopword
        else:
            feats[idx] = [0] #is not stopword
    return feats
