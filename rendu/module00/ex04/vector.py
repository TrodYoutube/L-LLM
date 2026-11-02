class Vector:
    def __init__(self, values: list[int]):
        val = []
        for x in values:
            val.append(float(x))
        self.val = val

    def __len__(self):
        return (len(self.val))

    def __getitem__(self, i):
        return (self.val[i])
    
    def __add__(self, other):
        tab = []
        if (len(self.val) == len(other)):
            for x in range(len(self.val)):
                tab.append(self.val[x] + other[x])
            return (Vector(tab))
        else:
            raise ValueError("Size Error")
    
    def __sub__(self, other):
        tab = []
        if (len(self.val) == len(other)):
            for x in range(len(self.val)):
                tab.append(self.val[x] - other[x])
            return (Vector(tab))
        else:
            raise ValueError("Size Error")

    def __mul__(self, other):
        tab = []
        for x in range(len(self.val)):
            tab.append(self.val[x] * other)
        return (Vector(tab))

    def __rmul__(self, other):
        return (self.__mul__(other))
    
    def __eq__(self, other):
        if (not isinstance(other, Vector)):
            return (NotImplemented)
        if (self.val == other.val):
            return (True)
        return (False)
    
    def dot(self, other):
        if (len(self.val) != len(other)):
            raise ValueError("Dot Product Error")
        total = 0
        for i in range(len(self.val)):
            total += self.val[i] * other[i]
        return (total)
    
    def norm(self):
        return (self.dot(self) ** 0.5)
    
    def __repr__(self):
        return (f"Vector({self.val})")