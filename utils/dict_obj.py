# Imports
from typing import TypeVar, Generic, Dict


# Define type variables for key and value
K = TypeVar("K")
V = TypeVar("V")


class DictObj(Dict[K, V], Generic[K, V]):
    """Generic dictionary object representation."""

    def __init__(self, *args, **kwargs):
        """Constructor of the DictObj class"""
        super(DictObj, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DictObj(value)

    def __getattr__(self, name: K) -> V:
        """Get the value of the reference key

        Args:
            name (K): Name of the key

        Returns:
            V: Value of the key
        """
        return self.get(name, None)

    def __setattr__(self, name: K, val: V) -> None:
        """Set or create the value of the key

        Args:
            name (K): Name of the key
            val (V): Value of the key
        """
        self[name] = val

    def __delattr__(self, name: K) -> None:
        """Delete the key from the dictionary

        Args:
            name (K): Name of the key
        """
        del self[name]
