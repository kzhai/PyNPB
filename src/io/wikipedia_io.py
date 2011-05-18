from glob import glob;
from collections import defaultdict;
from nltk.probability import FreqDist;
from string import rstrip;

# this method reads in the data from wikipedia dataset/corpus
# output a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
def parse_wikipedia(glob_expression, lang="english", doc_limit= -1):
    import codecs
    include_id_url = False
    
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

    #filtered_words = cutoff_df_de_news(glob_expression, doc_limit, max_df_percentage, min_df_percentage)

    from string import ascii_lowercase
    docs = {}
        
    dirs = glob(glob_expression)
    for jj in dirs:
        files = glob(jj + "/*")
        print("Found %i files in directory %s" % (len(files), jj))
        
        for ii in files:
            text = open(ii, mode="w", encoding="utf-8").lower()
            sections = text.split("\n</doc>\n")
            
            for section in sections:
                if section != None and len(section) != 0:
                    tokens = section.split("\n")
                    title = rstrip(tokens[1], ".")
                    contents = " ".join(tokens[2 :])
                                        
                    # not x in stop: to remove the stopwords
                    # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                    #words = [x for x in tokenizer.tokenize(contents) if (not x in stop) and (min(y in ascii_lowercase for y in x)) and (not x in filtered_words)]
                    words = [x for x in tokenizer.tokenize(contents) if (not x in stop) and (min(y in ascii_lowercase for y in x))]

                    if include_id_url:
                        docs["%s\t%s" % (tokens[0].strip(), title.strip())] = words
                    else:
                        docs["%s" % (title.strip())] = words
                
                if doc_limit > 0 and len(docs) > doc_limit:
                    print("Passed doc limit %i" % len(docs))
                    break
                
            if doc_limit > 0 and len(docs) > doc_limit:
                print("Passed doc limit %i" % len(docs))
                break
            
        if doc_limit > 0 and len(docs) > doc_limit:
            print("Passed doc limit %i" % len(docs))
            break
    
    return docs

# this method reads in the document mapping from wikipedia dataset/corpus
# output a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
def retrieve_doc_mappings(input_file):
    import codecs;
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    
    titles_a = []
    titles_b = []
    mapping_count=0;
    for line in input:
        titles = line.split("\t")
        if len(titles)!=2:
            print "error in parsing the titles: ", line
        
        titles_a.append(titles[0].strip().lower())
        titles_b.append(titles[1].strip().lower())
        mapping_count=mapping_count+1
                
    return titles_a, titles_b

def output_mapped_documents(titles, wiki_file, output_file):
    import codecs
    
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    total_docs = 0;
    loaded_docs = 0;
    input = codecs.open(wiki_file, mode="r", encoding="utf-8")

    for line in input:
        total_docs = total_docs+1
        contents = line.split("\t")
        contents[0] = contents[0].strip()
        contents[1] = contents[1].strip()

        if contents[0] in titles:
            loaded_docs = loaded_docs+1
            output.write(str(titles.index(contents[0])) + "\t" + contents[1].strip() + "\n")
            
        if total_docs%10000==0:
            print "output " + str(loaded_docs) + " mapped documents from " + str(total_docs) + " documents..."

    print "successfully output " + str(loaded_docs) + " mapped documents from " + str(total_docs) + " documents..."

"""

"""
def output_document_titles(wiki_file, output_file):
    import codecs
    
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    total_docs = 0;
    loaded_docs = 0;
    input = codecs.open(wiki_file, mode="r", encoding="utf-8")

    for line in input:
        total_docs = total_docs+1
        contents = line.split("\t")
        contents[0] = contents[0].strip()
        contents[1] = contents[1].strip()

        output.write(contents[0] + "\n");
        
    print "successfully output " + str(loaded_docs) + " mapped documents from " + str(total_docs) + " documents..."

"""

"""
def output_document_mappings(mapping_file_a, mapping_file_b, output_file, memory_efficient=False, document_window = 100000):
    import codecs

    #title_a, title_b = retrieve_doc_mappings(title_mapping_file)
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    
    if memory_efficient:
        parsed_docs=0;
        while True:
            output_docs = {}
            loaded_docs = 0;
            input = codecs.open(mapping_file_a, mode="r", encoding="utf-8")
            for line in input:
                loaded_docs = loaded_docs+1;
                if loaded_docs<=parsed_docs:
                    continue;
                
                contents = line.split("\t")
                contents[0] = contents[0].strip()
                contents[1] = contents[1].strip()
                output_docs[contents[0]] = contents[1].strip()

                if loaded_docs%10000==0:
                    print "load another 10000 documents..."
                if loaded_docs==parsed_docs+document_window:
                    break
            print "successfully load " + str(loaded_docs-parsed_docs) + " mapped documents..."
            
            if loaded_docs-parsed_docs==0:
                break;
            parsed_docs = loaded_docs;
            
            loaded_docs = 0;
            input = codecs.open(mapping_file_b, mode="r", encoding="utf-8")
            for line in input:
                contents = line.split("\t")
                contents[0] = contents[0].strip()
                contents[1] = contents[1].strip()
                
                loaded_docs = loaded_docs+1
                
                if contents[0] in output_docs.keys():
                    output.write(output_docs[contents[0]] + "\t" + contents[1].strip() + "\n")
                            
                if loaded_docs%10000==0:
                    print "load " + str(loaded_docs) + " mapped documents..."
            print "successfully load " + str(loaded_docs) + " mapped documents..."
    else:
        output_docs = {}
        loaded_docs = 0
        input = codecs.open(mapping_file_a, mode="r", encoding="utf-8")
        for line in input:
            contents = line.split("\t")
            contents[0] = contents[0].strip()
            contents[1] = contents[1].strip()
            
            loaded_docs = loaded_docs+1
            #output.write(str(titles_a.index(contents[0])) + "\t" + contents[1].strip())
            output_docs[contents[0]] = contents[1].strip()
                
            if loaded_docs%10000==0:
                print "load " + str(loaded_docs) + " mapped documents..."    
        print "successfully load " + str(loaded_docs) + " mapped documents..."
        
        loaded_docs = 0;
        input = codecs.open(mapping_file_b, mode="r", encoding="utf-8")
        for line in input:
            contents = line.split("\t")
            contents[0] = contents[0].strip()
            contents[1] = contents[1].strip()
            
            loaded_docs = loaded_docs+1
            
            if contents[0] not in output_docs.keys():
                print "warning: document mapping for index " + contents[0] + " not found..."
            else:
                output.write(output_docs[contents[0]] + "\t" + contents[1].strip() + "\n")
                        
            if loaded_docs%10000==0:
                print "load " + str(loaded_docs) + " mapped documents..."
        print "successfully load " + str(loaded_docs) + " mapped documents..."
    
    print "successfully output the document mappings..."
    
"""

"""

# this method reads in the document mapping from wikipedia dataset/corpus
# output a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
def output_title_mappings(input_file, output_file, lang="german"):
    prefix = "de:"
    if lang.lower() == "german":
        prefix = "de:"
    else:
        print "language option unspecified, default to german..."

    import codecs;
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    
    mapping_count=0;
    for line in input:
        titles = line.split("\t")
        if len(titles)!=2:
            print "error in parsing the titles: ", line
        
        if titles[1].startswith(prefix):
            output.write(titles[0].strip() + "\t" + titles[1].lstrip(prefix))
            mapping_count=mapping_count+1
        else:
            continue;
        
    return mapping_count

# this method reads in the data from wikipedia dataset/corpus
# output a dict data type, indexed by the document id, value is a list of the words in that document, not necessarily unique
def output_wikipedia(glob_expression, output_path, lang="english", doc_limit= -1):
    include_id_url = False
    
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

    #filtered_words = cutoff_df_de_news(glob_expression, doc_limit, max_df_percentage, min_df_percentage)

    from os import remove, access, F_OK;
    if access(output_path, F_OK):
        remove(output_path);

    from string import ascii_lowercase
    docs = {}

    parsed_docs = 0    
    dirs = glob(glob_expression)
    for jj in dirs:
        files = glob(jj + "/*")
        print("Found %i files in directory %s" % (len(files), jj))
        
        for ii in files:
            text = open(ii).read().lower()
            sections = text.split("\n</doc>\n")
            
            for section in sections:
                if section != None and len(section) != 0:
                    tokens = section.split("\n")
                    title = rstrip(tokens[1], ".")
                    contents = " ".join(tokens[2 :])
                                        
                    # not x in stop: to remove the stopwords
                    # min(y in ascii_lowercase for y in x) : to remove punctuation or any expression with punctuation and special symbols
                    #words = [x for x in tokenizer.tokenize(contents) if (not x in stop) and (min(y in ascii_lowercase for y in x)) and (not x in filtered_words)]
                    words = [x for x in tokenizer.tokenize(contents) if (not x in stop) and (min(y in ascii_lowercase for y in x))]

                    if include_id_url:
                        docs["%s\t%s" % (tokens[0].strip(), title.strip())] = words
                    else:
                        docs["%s" % (title.strip())] = words
                        
                    parsed_docs = parsed_docs+1

                if doc_limit > 0 and parsed_docs > doc_limit:
                    print("Passed doc limit %i" % len(docs))
                    break
                
            append(docs, output_path)
            docs.clear()
                 
            if doc_limit > 0 and parsed_docs > doc_limit:
                #print("Passed doc limit %i" % len(docs))
                break
            
        if doc_limit > 0 and parsed_docs > doc_limit:
            #print("Passed doc limit %i" % len(docs))
            break
        
    return parsed_docs
    
def append(docs, file):
    f = open(file, "a")
    
    for doc in docs.keys():
        f.write(doc)
        f.write("\t" + " ".join(docs[doc]))
        f.write("\n")

# this method convert a corpus into proper format for training lda model for variational inference
# output a defaultdict(dict) data type, first indexed by the document id, then indexed by the unique tokens
# corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
def parse_data(corpus):
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
    parsed_docs = output_wikipedia("../../data/wiki/enwiki/*", "../../data/wiki/enwiki.txt", "english", -1)
    print "parsed ", parsed_docs, " english documents in total..."
    
    parsed_docs = output_wikipedia("../../data/wiki/dewiki/*", "../../data/wiki/dewiki.txt", "german", -1)
    print "parsed ", parsed_docs, " german documents in total..."

#    output_document_titles("../../data/wiki/enwiki.txt", "../../data/wiki/en-title.txt")
#    output_document_titles("../../data/wiki/dewiki.txt", "../../data/wiki/de-title.txt")

    title_a, title_b = retrieve_doc_mappings("../../data/wiki/mapping-en-de/en-de-title-mapping.txt")
    print len(title_a), len(title_b)
    
    output_mapped_documents(title_a, "../../data/wiki/mapping-en-de/enwiki.txt", "../../data/wiki/mapping-en-de/en-mapping-wiki.txt")
    output_mapped_documents(title_b, "../../data/wiki/mapping-en-de/dewiki.txt", "../../data/wiki/mapping-en-de/de-mapping-wiki.txt")
    
    #output_document_mappings("../../data/wiki/mapping-en-de/en-doc.txt", "../../data/wiki/mapping-en-de/de-doc.txt", "../../data/wiki/en-de-doc.txt")
    output_document_mappings("en-doc.txt", "de-doc.txt", "en-de-doc.txt", True, 100000)
