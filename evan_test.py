from evan_feats import *
import pdb

wordarray = ['A','quick','and','dirty','test.']

print 'Testing:'
print wordarray

print 'stopword feature:'
print isstop(wordarray)

'''
print 'wordnet polysemy feats:'
print polysemy_feats(wordarray)
print "pretraining ppdb..."
ppdbtoks = ppdb_pretrain()
print ppdb_feats(ppdbtoks, wordarray)
'''
print "relative sentence position.."

print relsentpos(wordarray)

print "pretraining pmi collocations..."
bc, bwc, tc, twc = pmi_colloc_pretrain()
print pmi_feats(bc, bwc, tc, twc, wordarray)

pdb.set_trace()
