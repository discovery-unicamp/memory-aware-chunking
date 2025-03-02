__all__ = ["b_to_gb", "kb_to_gb"]


def b_to_gb(b_value: int) -> float:
    return b_value / (1024**3)


def kb_to_gb(kb_value: int) -> float:
    return kb_value / (1024**2)
