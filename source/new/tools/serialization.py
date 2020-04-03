import json
from typing import Generic, Dict, Any, TypeVar

T = TypeVar("T")


class JsonSerializable(Generic[T]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> T:
        """
        name_class = d["name_class"]
        name_module = d["name_module"]
        this_module = importlib.import_module(name_module)
        this_class = getattr(this_module, name_class)
        """
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """
        this_class = self.__class__
        d = {
            "name_class": this_class.__name__,
            "name_module": this_class.__module__,
        }
        """
        raise NotImplementedError()

    @staticmethod
    def load_from(path_name: str) -> T:
        with open(path_name, mode="r") as file:
            d = json.load(file)
            return JsonSerializable.from_dict(d)

    def save_as(self, path: str):
        with open(path, mode="w") as file:
            d = self.to_dict()
            json.dump(d, file, indent=2, sort_keys=True)