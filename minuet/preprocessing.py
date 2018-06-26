import re


def replace_numbers(w):
    return re.sub('[0-9]', '0', w)


def lower(w):
    return w.lower()


def assemble(*funs):
    """Helper function composition function."""
    def closure(x):
        for f in funs:
            x = f(x)
        return x
    return closure
