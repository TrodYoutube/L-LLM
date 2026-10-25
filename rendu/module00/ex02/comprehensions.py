
def squares(n):
    return [x * x for x in range(n)]


def evens(lst):
    return [x for x in lst if x % 2 == 0]

def flatten(lst):
    return [x for row in lst for x in row]

def word_lengths(sentence):
    return ({mot: len(mot) for mot in sentence.split()})

if __name__ == "__main__":
    print(squares(5))
    print(evens([1, 3, 2, 6, 5]))
    print(word_lengths("Bonjour Ca Va"))