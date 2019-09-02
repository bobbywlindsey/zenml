import re
from zenml.vector import Vector
from collections import Counter


class Text():
    def __init__(self, v:Vector) -> None:
        self.v = v

    def regex_parse(self, regex_exp:str) -> Vector:
        regex = re.compile(regex_exp)
        parse = regex.findall
        return self.v.apply(lambda x: parse(x))


    def word_counts(self, regex_exp:str) -> Vector:
        vector_of_arrays: Vector = self.regex_parse(regex_exp)
        return vector_of_arrays.apply(lambda x: Counter(x))
