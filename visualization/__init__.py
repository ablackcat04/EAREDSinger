"""Interactive visualization of DiffSinger variance features."""

__all__ = [
    "ExtractionConfig",
    "VarianceFeatures",
    "extract_variances",
    "write_interactive_visualization",
]


def __getattr__(name: str):
    if name in {"ExtractionConfig", "VarianceFeatures", "extract_variances"}:
        from . import extractor

        return getattr(extractor, name)
    if name == "write_interactive_visualization":
        from .render import write_interactive_visualization

        return write_interactive_visualization
    raise AttributeError(name)