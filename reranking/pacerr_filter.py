from typing import Dict, Type, Callable, List, Tuple

def boundary(iterable: List[Tuple], num=1):
    return iterable[:num] + iterable[-num:]

def testing(iterable: List[Tuple]):
    return iterable

filter_function_map = {
        "boundary": boundary,
        "testing": testing,
}
