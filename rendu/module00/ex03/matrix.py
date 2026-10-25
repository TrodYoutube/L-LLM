def shape(A):
    return (len(A), len(A[0]))

def mat_add(A, B):
    if (shape(A) != shape(B)):
        raise ValueError("Can't Add")
    row = shape(A)[0]
    col = shape(A)[1]
    table = [[0] * col for i in range(row)]
    for i in range(row):
        for x in range(col):
            table[i][x] += A[i][x]
            table[i][x] += B[i][x]
    return (table)

def transpose(A):
    row = shape(A)[0]
    col = shape(A)[1]
    table = [[0] * row for i in range (col)]
    for i in range (col):
        for x in range (row):
            table[i][x] = A[x][i]
    return (table)

def dot(u, v):
    if (len(u) != len(v)):
        raise ValueError("Dot Product Error")
    total = 0
    for i in range(len(u)):
        total += u[i] * v[i]
    return (total)

def mat_mul(A, B):
    if (shape(A)[1] != shape(B)[0]):
        raise ValueError("Matrix Product Error")
    b = transpose(B)
    col = shape(B)[1]
    row = shape(A)[0]
    table = [[0] * col for i in range(row)]
    for j in range(row):
        for i in range(col):
            u = A[j]
            v = b[i]
            table[j][i] = dot(u, v)
    return (table) 

if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    print(mat_mul(A, B))