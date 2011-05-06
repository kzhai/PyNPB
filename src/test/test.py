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

def lgammln(xx):
  """
  Returns the gamma function of xx.
  Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
  (Adapted from: Numerical Recipies in C.)
  
  Usage:   lgammln(xx)
  
  Copied from stats.py by strang@nmr.mgh.harvard.edu
  """
  
  from math import log, exp

  coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516,
           0.120858003e-2, -0.536382e-5]
  x = xx - 1.0
  tmp = x + 5.5
  tmp = tmp - (x+0.5)*log(tmp)
  ser = 1.0
  for j in range(len(coeff)):
      x = x + 1
      ser = ser + coeff[j]/x
  return -tmp + log(2.50662827465*ser)


if __name__ == "__main__":
#     output_mapped_documents("/windows/d/Data/en-mapping-wiki.txt", "/windows/d/Data/en-mapping-wiki2.txt")
#     output_mapped_documents("/windows/d/Data/de-mapping-wiki.txt", "/windows/d/Data/de-mapping-wiki2.txt")
    from scipy.special import gamma, psi, gammaln, polygamma
    print lgammln(10), gammaln(10)