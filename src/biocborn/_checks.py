from typing import Any, Callable

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def is_list_of_type(x: Any, target_type: Callable) -> bool:
    """Checks if `x` is a list of `target_type`.

    Args:
        x (Any): Any object.
        target_type (Callable): Type to check for, e.g. str, int.

    Returns:
        bool: True if 'x' is list and all values are of the same type.
    """
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        isinstance(item, target_type) for item in x
    )
