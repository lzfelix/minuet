import re


def replace_numbers(w):
    return re.sub('[0-9]', '0', w)


def lower(w):
    return w.lower()


def assemble(*funs):
    """Helper function to assemble a preprocessing pipeline. Example:
    `pre = assemple(lower, replace_numbers)`.

    :param functions that will compose the pipeline
    :return a sequential pipeline function to be used by Minuet
    ."""

    def closure(x):
        for f in funs:
            x = f(x)
        return x
    return closure
