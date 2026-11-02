def mean(lst):
    if len(lst) == 0:
        raise ValueError("Empty List")
    total = 0
    for x in lst:
        total += x
    return (total / len(lst))

def variance(lst):
    total = 0
    meant = mean(lst)
    for x in lst:
        total += (x - meant)**2
    return (total / len(lst))

def std(lst):
    return (variance(lst) ** 0.5)

def standardize(lst):
    tab = []
    meant = mean(lst)
    stdt = std(lst)
    for x in lst:
        tab.append((x - meant) / stdt)
    return (tab)

def load_column(path, col):
    tab = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        if (col not in header):
            raise KeyError("Key Error")
        i = header.index(col)
        for line in f:
            champs = line.split(",")
            tab.append(float(champs[i]))
    return (tab)
     
