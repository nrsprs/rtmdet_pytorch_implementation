from typing import overload


@overload
def apply_factor(values: int, factor: float) -> int: ...
@overload
def apply_factor(values: list[int], factor: float) -> list[int]: ...


def apply_factor(values, factor: float):
    def _scale(v: int) -> int:
        return max(1, round(v * factor))

    if isinstance(values, int):
        return _scale(values)
    else:
        return [_scale(v) for v in values]
