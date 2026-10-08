from typing import Any


class ModelUnavailableError(LookupError):
    pass


class ModelRegistry:
    """In-memory contract; persistence/loading adapters can be added later."""
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], Any] = {}

    def register(self, name: str, version: str, model: Any) -> None:
        if not name or not version:
            raise ValueError("Model name and version are required.")
        self._models[(name, version)] = model

    def load(self, name: str, version: str) -> Any:
        try:
            return self._models[(name, version)]
        except KeyError as error:
            raise ModelUnavailableError(f"Model {name!r} version {version!r} is not registered.") from error

    def available(self) -> list[tuple[str, str]]:
        return sorted(self._models)
