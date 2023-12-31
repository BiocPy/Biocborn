from typing import Any, Callable

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def is_list_of_type(x: Any, target_type: Callable) -> bool:
    """Checks if ``x`` is a list, and whether all elements of the list are of the same type.

    Args:
        x (Any): Any list.
        target_type (callable): Type to check for,
            e.g. :py:class:`str`, :py:class:`int`.

    Returns:
        bool: True if ``x`` satisfies the conditions.
    """
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        isinstance(item, target_type) for item in x
    )
