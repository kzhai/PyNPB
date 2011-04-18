def output_mapped_documents(input_file, output_file):
    import codecs, string
    
    output = codecs.open(output_file, mode="w", encoding="utf-8")
    total_docs = 0;
    loaded_docs = 0;
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    for line in input:
        docs = line.split("\t")
        
        new_id = docs[0]
        for data in docs[1:]:
            id = new_id
            
            contents = data.split(" ")
            
            for i in xrange(len(contents[-1])):
                if contents[-1][i] in string.digits:
                    index=i
                    break
            new_id = contents[-1][index:]
            contents[-1] = contents[-1][:index]
            
            output.write(id + "\t" + " ".join(contents) + "\n")

    print "successfully output " + str(loaded_docs) + " mapped documents from " + str(total_docs) + " documents..."

if __name__ == "__main__":
     output_mapped_documents("/windows/d/Data/en-mapping-wiki.txt", "/windows/d/Data/en-mapping-wiki2.txt")
     output_mapped_documents("/windows/d/Data/de-mapping-wiki.txt", "/windows/d/Data/de-mapping-wiki2.txt")