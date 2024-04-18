class Criterion:
    def __init__(self, name, type, weight, q, p, veto=None):
        self.name = name
        self.type = type
        self.weight = weight
        self.q = q
        self.p = p
        self.veto = veto