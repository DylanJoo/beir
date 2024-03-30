from collections import defaultdict
from typing import Dict, Type, Callable, List, Tuple

def boundary(iterable: List[Tuple], num=1):
    return iterable[:num] + iterable[-num:]

def top(iterable: List[Tuple], num=1):
    return iterable[:num]

def bottom(iterable: List[Tuple], num=1):
    return iterable[:1] + iterable[-num:]

def top_bottom(iterable: List[Tuple], n1=1, n2=1):
    return iterable[:n1] + iterable[-n2:]

def identity(iterable: List[Tuple], **kwargs):
    return iterable

filter_function_map = defaultdict(lambda: identity)
filter_function_map["boundary"] = boundary
filter_function_map["top"] = top
filter_function_map["bottom"] = bottom
filter_function_map["top_bottom"] = top_bottom
