from glob import glob;
from collections import defaultdict;
from nltk.probability import FreqDist;

def parse_de_news_gs(glob_expression, lang="english", doc_limit= -1, max_df_percentage = 1.0, min_df_percentage = 0.0):
    docs = parse_de_news(glob_expression, lang, doc_limit, max_df_percentage, min_df_percentage);
    return docs

def parse_de_news_vi(glob_expression, lang="english", doc_limit= -1, max_df_percentage = 1.0, min_df_percentage = 0.0):
    docs = parse_de_news(glob_expression, lang, doc_limit, max_df_percentage, min_df_percentage);
    return convert_format_gs2vi(docs)

# compute the df counts of the given corpus
# max_df_percentage: a value between 0 to 1, upper cutoff for df is computed as document number times max_df_percentage
# min_df_percentage: a value between 0 to 1, lower cutoff for df is computed as document number times min_df_percentage
def cutoff_df_de_news(glob_expression, doc_limit= -1, max_df_percentage = 1.0, min_df_percentage = 0.0):
    from nltk.tokenize.treebank import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()

    from string import ascii_lowercase
    files = glob(glob_expression)
    print("Found %i files" % len(files))
    
    df = FreqDist()
    doc_count = 0
    
    for ii in files:
        text = open(ii).read().lower()
        
        sections = text.split("<doc")
        
        for section in sections:
            if section != None and len(section) != 0:
                index_content = section.split(">\n<h1>\n")
                title_content = index_content[1].split("</h1>")
                # not x in stop: to remove the stopwords
                # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                words = [x for x in tokenizer.tokenize(title_content[1]) if min(y in ascii_lowercase for y in x)]
    
                words = set(words)
                for word in words:
                    df.inc(word, 1)
                    
                doc_count += 1
                    
        if doc_limit > 0 and doc_count>doc_limit:
            print("Passed doc limit %i" % doc_count)
            break
    
    max_df = (doc_count*max_df_percentage)
    min_df = (doc_count*min_df_percentage)
    filtered_words = [word for word in df.keys() if df[word]>max_df or df[word]<min_df]
    print("document count %i,\tupper df cutoff %f,\tlower df cutoff %f" % (doc_count, max_df, min_df))
    print "filtered words: ", filtered_words

    return filtered_words

# this method reads in the data from de-news dataset/corpus
# output a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
# this format is generally used for gibbs sampling
def parse_de_news(glob_expression, lang="english", doc_limit= -1, max_df_percentage = 1.0, min_df_percentage = 0.0):
    include_title = False
    include_path = False
    
    from nltk.tokenize.treebank import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()

    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    if lang.lower() == "english":
        stop = stopwords.words('english')
    elif lang.lower() == "german":
        stop = stopwords.words('german')
    else:
        print "language option unspecified, default to english..."

    filtered_words = cutoff_df_de_news(glob_expression, doc_limit, max_df_percentage, min_df_percentage)

    from string import ascii_lowercase
    docs = {}
    files = glob(glob_expression)
    print("Found %i files" % len(files))
    
    for ii in files:
        text = open(ii).read().lower()
        
        sections = text.split("<doc")
        
        for section in sections:
            if section != None and len(section) != 0:
                index_content = section.split(">\n<h1>\n")
                title_content = index_content[1].split("</h1>")
                # not x in stop: to remove the stopwords
                # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                words = [x for x in tokenizer.tokenize(title_content[1]) if (not x in stop) and (min(y in ascii_lowercase for y in x)) and (not x in filtered_words)]
                if include_path:
                    if include_title:
                        docs["%s\t%s\t%s" % (ii, index_content[0].strip(), title_content[0].strip())] = words
                    else:
                        docs["%s\t%s" % (ii, index_content[0].strip())] = words
                else:
                    if include_title:
                        docs["%s\t%s" % (index_content[0].strip(), title_content[0].strip())] = words
                    else:
                        docs["%s" % (index_content[0].strip())] = words
                                        
        if doc_limit > 0 and len(docs) > doc_limit:
            print("Passed doc limit %i" % len(docs))
            break
    
    return docs

# this method convert a corpus from gibbs sampling format into proper format for variational inference format
# output a defaultdict(dict) data type, first indexed by the document id, then indexed by the unique tokens
# corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
def convert_format_gs2vi(corpus):
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

if __name__ == "__main__":
    data_en = parse_de_news("/windows/d/Data/de-news/txt/*.en.txt", "english",
                  1, 0.4, 0.0001)
    #print data_en
    
    data_en = convert_format_gs2vi(data_en)
    data_de = parse_de_news("/windows/d/Data/de-news/txt/*.de.txt", "german",
                  1, 0.4, 0.0001)
    data_de = convert_format_gs2vi(data_de)
    print len(data_en), "\t", len(data_de)
    
    from io.io_adapter import map_corpus
    [data_en, data_de] = map_corpus(data_en, data_de)
    print len(data_en), "\t", len(data_de)
    
#lda.initialize(d)

#lda.sample(100)
#lda.print_topics()