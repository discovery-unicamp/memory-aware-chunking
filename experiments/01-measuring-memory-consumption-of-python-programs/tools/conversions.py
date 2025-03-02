__all__ = ["kb_to_gb"]


def kb_to_gb(kb_value: int) -> float:
    return kb_value / (1024**2)
