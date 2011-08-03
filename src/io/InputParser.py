def import_monolingual_data(input_file, doc_limit=-1):
    import codecs
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    
    doc_count = 0
    docs = {}
    
    for line in input:
        line = line.strip().lower();

        contents = line.split("\t");
        assert(len(contents)==2);
        docs[contents[0]] = contents[1].split()
        
        doc_count+=1 
    
        if doc_count%10000==0:
            print "successfully import " + str(doc_count) + " documents..."
            
        if doc_limit > 0 and doc_count > doc_limit:
            print("Passed doc limit %i" % doc_count)
            return docs
    
    print "successfully import all documents..."
    return docs
