from collections import defaultdict
from typing import Dict, Type, Callable, List, Tuple

def boundary(iterable: List[Tuple], num=1):
    return iterable[:num] + iterable[-num:]

def top(iterable: List[Tuple], num=1):
    return iterable[:num]

def bottom(iterable: List[Tuple], num=1):
    return iterable[-num:]

def identity(iterable: List[Tuple], **kwargs):
    return iterable

filter_function_map = defaultdict(lambda: identity)
filter_function_map["boundary"] = boundary
filter_function_map["top"] = top
filter_function_map["bottom"] = bottom
