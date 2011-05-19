from glob import glob;
from collections import defaultdict;

def output_param(alpha, beta, gamma, dir, index=-1):
    if index!=-1:
        postfix = str(index)
    else:
        postfix = ""

    alpha_path = dir + "alpha" + postfix
    f = open(alpha_path, "w");
    for k in alpha.keys():
        f.write(str(k) + "\t" + str(alpha[k]) + "\n")
        
    beta_path = dir + "beta" + postfix
    f = open(beta_path, "w");
    for term in beta.keys():
        for k in beta[term].keys():
            f.write(str(term) + "\t" + str(k) + "\t" + str(beta[term][k]) + "\n")
        
    gamma_path = dir + "gamma" + postfix
    f = open(gamma_path, "w");
    for doc in gamma.keys():
        for k in gamma[doc].keys():
            f.write(str(doc) + "\t" + str(k) + "\t" + str(gamma[doc][k]) + "\n")
            
def output(data, file):
    f = open(file, "w");
    
    terms = [];
    for value in data.values():
        terms = terms+value;
    terms = set(terms);
    
    termID = {}
    i = 0;
    for t in terms:
        termID[t] = i;
        i+=1;
    
    i = 1;
    for doc in data.keys():
        if len(data[doc])==0:
            continue;
        f.write(str(i) + "\t");
        temp = set(data[doc]);
        for t in temp:
            f.write(str(termID[t])+ "\t"+str(data[doc].count(t))+"\t");
        f.write("\n");
        i+=1;
        
def append(data, file):
    f = open(file, "a");
    
    terms = [];
    for value in data.values():
        terms = terms+value;
    terms = set(terms);
    
    termID = {}
    i = 0;
    for t in terms:
        termID[t] = i;
        i+=1;
    
    i = 1;
    for doc in data.keys():
        if len(data[doc])==0:
            continue;
        f.write(str(i) + "\t");
        temp = set(data[doc]);
        for t in temp:
            f.write(str(termID[t])+ "\t"+str(data[doc].count(t))+"\t");
        f.write("\n");
        i+=1;
        
if __name__ == "__main__":
    from io.de_news_io import parse_de_news_vi, map_corpus
    
    data_en = parse_de_news_vi("../../data/de-news/txt/*.en.txt", "english",
                  1500, 0.2, 0.0001)
    
    data_de = parse_de_news_vi("../../data/de-news/txt/*.de.txt", "german",
                  1500, 0.2, 0.0001)
    print len(data_en), "\t", len(data_de)
    
    [data_en, data_de] = map_corpus(data_en, data_de)
    
    print data_en.keys()
    print data_de.keys()
    print len(data_en), len(data_de)        