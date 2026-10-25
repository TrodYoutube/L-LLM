def ft_sum(lst):
    total = 0
    for nombre in lst:
        total += nombre
    return (total)

def ft_max(lst):
    if (len(lst) == 0):
        raise ValueError("Liste Vide")
    Max = lst[0]
    for i in range(len(lst)):
        if (lst[i] > Max):
            Max = lst[i]
    return (Max)
        
def count_chars(s):
    d ={}
    for c in s:
        if c in d:
            d[c] += 1
        else:
            d[c] = 1
    return (d)

if __name__ == "__main__":
    print(ft_sum([1, 5]))
    print(count_chars("hello"))
    print(count_chars(""))
    print(ft_max([-5, -2, -9]))