from glob import glob;
from collections import defaultdict;

def output_defaultdict_dict(defaultdict_dict, dir, title, index=-1):
    if index!=-1:
        postfix = str(index)
    else:
        postfix = ""
     
    py_defaultdict_dict = dir + title + postfix
    f = open(py_defaultdict_dict, "w");
    for doc in defaultdict_dict.keys():
        for k in defaultdict_dict[doc].keys():
            f.write(str(doc) + "\t" + str(k) + "\t" + str(defaultdict_dict[doc][k]) + "\n")
            
def output_dict(py_dict, dir, title, index=-1):
    if index!=-1:
        postfix = str(index)
    else:
        postfix = ""

    py_dict_path = dir + title + postfix
    f = open(py_dict_path, "w");
    for k in py_dict.keys():
        f.write(str(k) + "\t" + str(py_dict[k]) + "\n")

if __name__ == "__main__":
    from io.de_news_io import parse_de_news
    
    data_en = parse_de_news("../../data/de-news/*.en.txt", "english", 1, 0.04, 1)
#    data_en = list2dict(data_en)
    
    data_de = parse_de_news("/windows/d/Data/de-news/txt/*.de.txt", "german",
                  1, False)
#    data_de = list2dict(data_de)
    
    alpha = {};
    alpha[1] = 14.53
    alpha[2] = 58391.510
    alpha[4]= 395.5390185
        
    print "synch point", data_en.keys(), data_de.keys()
    
#lda.initialize(d)

#lda.sample(100)
#lda.print_topics()