from glob import glob;
from collections import defaultdict;
from nltk.probability import FreqDist;
from string import rstrip;

from MonolingualIOParser import MonolingualIOParser;

class WikipediaParser(MonolingualIOParser):
    """
    @param max_df_percentage: a value between 0 to 1, upper cutoff for document frequency is computed as document number times max_df_percentage
    @param min_df_percentage: a value between 0 to 1, lower cutoff for document frequency is computed as document number times min_df_percentage
    """
    def __init__(self):
        MonolingualIOParser.__init__(self, 0.5, 0.00005);
    
    def tokenize_data(self, glob_expression, doc_limit=-1, filtered_words=[]):
        import codecs
        
        import string
        exclude = set(string.punctuation);
        
        from nltk.tokenize.punkt import PunktWordTokenizer
        tokenizer = PunktWordTokenizer()
        #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
        parsed_docs = 0
        docs = {}
        dirs = glob(glob_expression)
        for jj in dirs:
            files = glob(jj + "/*")
            print("Found %i files in directory %s" % (len(files), jj))
            
            for ii in files:
                text = codecs.open(ii, mode="r", encoding="utf-8")
                
                title_flag = False;
                file_flag = False;
                title = None;
                for line in text:
                    line = line.strip().lower();
    
                    if len(line)==0:
                        continue;
                    
                    if line.startswith("<doc"):
                        assert(not file_flag);
                        assert(not title_flag);
                        title_flag = True;
                        file_flag = True;
                        contents = []
                    elif line.startswith("</doc>"):
                        assert(file_flag);
                        assert(not title_flag);
                        file_flag = False;
                        docs["%s" % (title)] = contents;
                        parsed_docs = parsed_docs+1;
                        
                        if parsed_docs%100000==0:
                            print "successfully load " + str(parsed_docs) + " documents..."
                            
                        if doc_limit > 0 and parsed_docs > doc_limit:
                            print("Passed doc limit %i" % parsed_docs)
                            return docs
                    else:
                        assert(file_flag)
                        
                        # remove all punctuations from the input
                        line = ''.join(ch for ch in line if ch not in exclude);
    
                        if title_flag:
                            title_flag = False;
                            title = line;
                        else:
                            # not x in stop: to remove the stopwords
                            # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                            #words = [x for x in tokenizer.tokenize(contents) if (not x in stop) and (min(y in ascii_lowercase for y in x)) and (not x in filtered_words)]
                            words = [x for x in tokenizer.tokenize(line) if (not x in filtered_words)]
                            contents = contents+words

        return docs
    
    """
    """
    def export_data(self, glob_expression, output_file, doc_limit=-1, lang='english'):
        import codecs
        
        import string
        exclude = set(string.punctuation);
        
        from nltk.tokenize.punkt import PunktWordTokenizer
        tokenizer = PunktWordTokenizer()
        #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        if lang.lower() == "english":
            stop = stopwords.words('english')
        elif lang.lower() == "german":
            stop = stopwords.words('german')
        else:
            print "language option unspecified, default to english..."
            
        output = codecs.open(output_file, mode="w", encoding="utf-8")
    
        parsed_docs = 0
        dirs = glob(glob_expression)
        for jj in dirs:
            files = glob(jj + "/*")
            print("Found %i files in directory %s" % (len(files), jj))
            
            for ii in files:
                text = codecs.open(ii, mode="r", encoding="utf-8")
                
                title_flag = False;
                file_flag = False;
                title = None;
                for line in text:
                    line = line.strip().lower();
    
                    if len(line)==0:
                        continue;
                    
                    if line.startswith("<doc"):
                        assert(not file_flag);
                        assert(not title_flag);
                        title_flag = True;
                        file_flag = True;
                        contents = []
                    elif line.startswith("</doc>"):
                        assert(file_flag);
                        assert(not title_flag);
                        file_flag = False;
                        output.write(title + "\t" + " ".join(contents) + "\n")
                        parsed_docs = parsed_docs+1;
                        
                        if parsed_docs%100000==0:
                            print "successfully load " + str(parsed_docs) + " documents..."
                            
                        if doc_limit > 0 and parsed_docs > doc_limit:
                            print("Passed doc limit %i" % parsed_docs)
                            return parsed_docs
                    else:
                        assert(file_flag)
                        
                        # remove all punctuations from the input
                        line = ''.join(ch for ch in line if ch not in exclude);
    
                        if title_flag:
                            title_flag = False;
                            title = line;
                        else:
                            # not x in stop: to remove the stopwords
                            # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                            #words = [x for x in tokenizer.tokenize(contents) if (not x in stop) and (min(y in ascii_lowercase for y in x)) and (not x in filtered_words)]
                            words = [x for x in tokenizer.tokenize(line) if (not x in stop)]
                            contents = contents+words

        return parsed_docs
    
if __name__ == "__main__":
    #from util.type_converter import dict_list_2_dict_freqdist
    dn_parser = WikipediaParser();
    dn_parser.export_data("../../data/wiki/enwiki/*", "../../data/wiki/en-wiki.txt", -1, "english");
    
    dn_parser = WikipediaParser();
    dn_parser.export_data("../../data/wiki/dewiki/*", "../../data/wiki/de-wiki.txt", -1, "german");
