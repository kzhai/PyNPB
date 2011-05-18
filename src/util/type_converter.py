# @author: ke zhai (zhaike@cs.umd.edu)

"""
this method convert a corpus from dict(list) to defaultdict(dict) format, similar words are grouped
@return: a defaultdict(dict) data type, first indexed by the document id, then indexed by the unique tokens
@param corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
"""
def dict_list_2_dict_dict(corpus):
    from collections import defaultdict;

    docs = defaultdict(dict)
    
    for doc in corpus.keys():
        content = {}
        for term in corpus[doc]:
            if term in content.keys():
                content[term] = content[term] + 1
            else:
                content[term] = 1
        docs[doc] = content
    
    return docs

"""
this method convert a corpus from dict(list) to defaultdict(FreqDist) format, similar words are grouped
@return: a defaultdict(FreqDist) data type, first indexed by the document id, then indexed by the unique tokens
@param corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
"""
def dict_list_2_dict_freqdist(corpus):
    from collections import defaultdict;
    from nltk.probability import FreqDist;

    docs = defaultdict(dict)
    
    for doc in corpus.keys():
        docs[doc] = FreqDist(corpus[doc])
    
    return docs

"""
this method convert two lists to a dict, two input lists must share the same length
@return: a dict data type, keyed by the elements in list_a and valued by the elements in list_b
@param list_a: a list contains the corresponding keys
@param list_b: a list contains the corresponding values
"""
def two_lists_2_dict(list_a, list_b):
    assert len(list_a)==len(list_b)
    
    return dict(zip(list_a, list_b))
