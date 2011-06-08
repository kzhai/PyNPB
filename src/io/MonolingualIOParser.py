from glob import glob;

class MonolingualIOParser:
    """
    @param max_df_percentage: a value between 0 to 1, upper cutoff for document frequency is computed as document number times max_df_percentage
    @param min_df_percentage: a value between 0 to 1, lower cutoff for document frequency is computed as document number times min_df_percentage
    """
    def __init__(self, max_df_percentage=1.0, min_df_percentage=0, include_path=False):
        self.token_delimiter = " ";
        self.cont_delimiter = "\t";
        self.doc_delimiter = "\n";
        
        self.max_df_percentage = max_df_percentage;
        self.min_df_percentage = min_df_percentage;
        
        self.include_path = include_path;
        
#    """    
#    """
#    @staticmethod
#    def parse_to_gs_format(self, glob_expression, lang="english", doc_limit= -1, ):
#        docs = self.parse_data(glob_expression, lang, doc_limit);
#        return docs
#    
#    """
#    """
#    @staticmethod
#    def parse_to_vi_format(self, glob_expression, lang="english", doc_limit= -1):
#        from util.type_converter import dict_list_2_dict_freqdist
#        docs = self.parse_data(glob_expression, lang, doc_limit);
#        return dict_list_2_dict_freqdist(docs)
    
    """
    compute the document frequency counts of the given corpus
    @return: a set data type contains the cut-off words by the document frequency range
    """
    def cutoff_words_by_df(self, glob_expression, doc_limit=-1):
        from nltk.probability import FreqDist;
        df = FreqDist()
                  
        docs = self.tokenize_data(glob_expression, doc_limit);
        
        doc_count = 0
        for doc in docs.keys():
            doc_count+=1
            words = set(docs[doc])
            for word in words:
                df.inc(word, 1)
        
        max_df = (doc_count*self.max_df_percentage)
        min_df = (doc_count*self.min_df_percentage)
        filtered_words = set([word for word in df.keys() if df[word]>max_df or df[word]<min_df])
        print("document count %i,\tupper df cutoff %f,\tlower df cutoff %f" % (doc_count, max_df, min_df))
        print "filtered words: ", filtered_words

        return filtered_words
    
    """
    this method reads in the data from de-news dataset/corpus
    @return: a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
    this format is generally used for gibbs sampling
    """
    def parse_data(self, glob_expression, doc_limit= -1, lang="english"):
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        if lang.lower() == "english":
            stop = stopwords.words('english')
        elif lang.lower() == "german":
            stop = stopwords.words('german')
        else:
            print "language option unspecified, default to english..."
    
        filtered_words = self.cutoff_words_by_df(glob_expression, doc_limit)
        docs = self.tokenize_data(glob_expression, doc_limit, set(list(filtered_words)+stop))
        
        return docs

    """
    this method reads in the data from de-news dataset/corpus
    output a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
    this format is generally used for gibbs sampling
    """
    def tokenize_data(self, glob_expression, doc_limit= -1, filtered_words=[]):
        import codecs
        
        print "parent tokenize method..."
        from nltk.tokenize.punkt import PunktWordTokenizer
        tokenizer = PunktWordTokenizer()
    
        from string import ascii_lowercase
        docs = {}
        files = glob(glob_expression)
        print("Found %i files" % len(files))
        
        for ii in files:
            text = codecs.open(ii, mode="r", encoding="utf-8")
            
            for line in text:
            
                contents = line.split(self.cont_delimiter);
                words = [x for x in tokenizer.tokenize(contents[1]) if not x in filtered_words]
                docs["%s" % (contents[0].strip())] = words
                                            
            if doc_limit > 0 and len(docs) > doc_limit:
                print("Passed doc limit %i" % len(docs))
                break
        
        return docs
    
    def export_data(self, glob_expression, output_file, doc_limit=-1, lang="english"):
        data = self.parse_data(glob_expression, doc_limit, lang)
        
        import codecs
        output = codecs.open(output_file, mode="w", encoding="utf-8")
        
        doc_count = 0
        for key in data.keys():
            output.write(key + self.cont_delimiter + self.token_delimiter.join(data[key]) + self.doc_delimiter)
            doc_count+=1
            if doc_count%10000==0:
                print "successfully export " + str(doc_count) + " documents..."
        
        print "successfully export all documents..."
            
    def import_data(self, input_file, doc_limit=-1):
        import codecs
        input = codecs.open(input_file, mode="r", encoding="utf-8")
        
        doc_count = 0
        docs = {}
        
        for line in input:
            line = line.strip().lower();

            contents = line.split(self.cont_delimiter);
            assert(len(contents)==2);
            docs[contents[0]] = contents[1]
            
            doc_count+=1 
        
            if doc_count%10000==0:
                print "successfully import " + str(doc_count) + " documents..."
                
            if doc_limit > 0 and doc_count > doc_limit:
                print("Passed doc limit %i" % doc_count)
                return docs
        
        print "successfully export all documents..."
        return docs

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

if __name__ == "__main__":
    from util.type_converter import dict_list_2_dict_freqdist
    
    data_en = parse_data("../../data/de-news/txt/*.en.txt", "english",
                  -1, 0.4, 0.0001)
    data_en = dict_list_2_dict_freqdist(data_en)
    data_de = parse_data("../../data/de-news/txt/*.de.txt", "german",
                  -1, 0.4, 0.0001)
    data_de = dict_list_2_dict_freqdist(data_de)
    print len(data_en), "\t", len(data_de)
    
    [data_en, data_de] = map_corpus(data_en, data_de)
    print len(data_en), "\t", len(data_de)