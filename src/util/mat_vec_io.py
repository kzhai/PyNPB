"""
@param path: export path
@param matrix: export matrix
"""
def export_matrix(path, matrix):
    output = open(path, "w");
    (row, col) = matrix.shape;
    for r in xrange(row):
        output.write(" ".join(["%s" % expression for expression in matrix[r, :]]) + "\n");

"""
@param path: export path
@param vector: export vector
"""
def export_vector(path, vector):
    output = open(path, "w");
    output.write(" ".join(["%s" % expression for expression in vector]) + "\n");

"""
@param path: import path
@param matrix: import matrix
"""
def import_matrix(path):
    input = open(path);
    matrix = None
    for line in input:
        row = line.split();
        row = [float(token) for token in row];
        if matrix==None:
            matrix = numpy.array([row]);
        else:
            assert(matrix.shape[1]==len(row));
            matrix = numpy.vstack((matrix, row));
    return matrix

"""
@param path: import path
@param vector: import vector
@attention: this method will only read in the first line of file.
"""
def import_vector(path):
    input = open(path);
    for line in input:
        row = line.split();
        row = [float(token) for token in row];
        vector = numpy.array(row);
        return vector
    
if __name__ == '__main__':
    A = import_matrix("../../output/tmp-output/A-matrix-10");
    print A.shape
    (a, b, c) = import_vector("../../output/tmp-output/Hyper-parameter-vector-10");
    print a, b, c
