import typing

# Suppress type-checking errors as a result of PyObjC's lazy import of the Quartz framework.
def __getattr__(name: str) -> typing.Any: ...
