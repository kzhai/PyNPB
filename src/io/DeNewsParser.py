from glob import glob;

from MonolingualIOParser import MonolingualIOParser

class DeNewsParser(MonolingualIOParser):
    """
    """
    def __init__(self, include_title=False):
        MonolingualIOParser.__init__(self);
        self.include_title = include_title
    
    """
    this method reads in the data from de-news dataset/corpus
    @return: a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
    this format is generally used for gibbs sampling
    """
    def tokenize_data(self, glob_expression, doc_limit= -1, filtered_words=[]):
        import codecs
    
        import string
        exclude = set(string.punctuation);
    
        from nltk.tokenize.punkt import PunktWordTokenizer
        tokenizer = PunktWordTokenizer()
    
        parsed_docs = 0
        docs = {}
        files = glob(glob_expression)
        print("Found %i files" % len(files))
        
        for ii in files:
            text = codecs.open(ii, mode="r", encoding="utf-8")
            
            title_flag = False;
            title = None;
            contents = []

            for line in text:
                line = line.strip().lower();

                if len(line)==0:
                    continue;
                
                if line.startswith("<doc"):
                    if title!=None:
                        docs["%s" % (title)] = contents;
                        #output.write(title + "\t" + " ".join(contents) + "\n");
                        parsed_docs = parsed_docs+1;
                        
                    contents = []
                    title = line.lstrip("<doc").rstrip(">").strip();
                elif line.startswith("<h1>"):
                    title_flag = True;
                elif line.startswith("</h1>"):
                    assert(title_flag);
                    title_flag = False;
                else:
                    if title_flag:
                        if self.include_title:
                            title = title + self.token_delimiter + line;
                    else:
                        # remove all punctuations from the input
                        line = "".join(ch for ch in line if ch not in exclude);
    
                        words = [x for x in tokenizer.tokenize(line) if (not x in filtered_words)]
                        contents = contents+words
            
                if doc_limit > 0 and parsed_docs > doc_limit:
                    print("Passed doc limit %i" % parsed_docs)
                    return docs
    
        return docs

if __name__ == "__main__":
    #from util.type_converter import dict_list_2_dict_freqdist
    dn_parser = DeNewsParser();
    dn_parser.export_data("../../data/de-news/txt/*.en.txt", "../../data/de-news/en-de-news.txt", -1, "english");

    
    dn_parser = DeNewsParser();
    dn_parser.export_data("../../data/de-news/txt/*.de.txt", "../../data/de-news/de-de-news.txt", -1, "german");
    
    '''
    data_en = parse_de_news("../../data/de-news/txt/*.en.txt", "english",
                  -1, 0.4, 0.0001)
    data_en = dict_list_2_dict_freqdist(data_en)
    data_de = parse_de_news("../../data/de-news/txt/*.de.txt", "german",
                  -1, 0.4, 0.0001)
    data_de = dict_list_2_dict_freqdist(data_de)
    print len(data_en), "\t", len(data_de)
    
    [data_en, data_de] = map_corpus(data_en, data_de)
    print len(data_en), "\t", len(data_de)
    '''


"""
this method is used for mapping documents correspondence between two corpora, usually for multilingual study, 
output the corresponding corpus with exact one-to-one mapping on the document id's

@param corpus_a: dict or defaultdict(dict) data type, indexed by document id
@param corpus_b: dict or defaultdict(dict) data type, indexed by document id
"""
def map_corpus(corpus_a, corpus_b):
    common_docs = (set(corpus_a.keys()) & set(corpus_b.keys()));
   
    for doc in corpus_a.keys():
        if doc not in common_docs:
            del corpus_a[doc]
            
    for doc in corpus_b.keys():
        if doc not in common_docs:
            del corpus_b[doc]
            
    return corpus_a, corpus_b

"""
this method outputs the mapped documents to an output file
"""
def output_mapped_documents(output_file, data_en, data_de):
    [data_en, data_de] = map_corpus(data_en, data_de)
    
    import codecs
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    
    for key in data_en.keys():
        output.write(data_en[key] + "\t" + data_de[key] + "\n")
                    
    print "successfully output the document mappings..."
    
def output_documents( data, output_file):
    import codecs
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    
    for key in data.keys():
        output.write(key + "\t" + " ".join(data[key]) + "\n")
    