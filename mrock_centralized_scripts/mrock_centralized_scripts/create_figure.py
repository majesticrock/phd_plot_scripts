import matplotlib.pyplot as plt

__GOLDEN_RATIO__ = (5.**.5 - 1.) / 2.
__DEFAULT_RATIO__ = 4.8 / 6.4

__NORMAL_TEXT_WIDTH_FRACTION__ = 0.5
__LARGE_TEXT_WIDTH_FRACTION__ = 0.99


def _load_text_width():
    import sys
    from pathlib import Path

    cwd = Path.cwd()
    cfg_path = cwd / "journal_text_width.py"

    if not cfg_path.exists():
        raise RuntimeError(
            f"journal_text_width.py not found in project root: {cwd}\n"
            "Create journal_text_width.py and define TEXT_WIDTH = <float>."
        )

    sys.path.insert(0, str(cwd))
    try:
        import journal_text_width
    except Exception as e:
        raise RuntimeError(
            "Failed to import journal_text_width.py.\n"
            "Check that the file is syntactically valid."
        ) from e
    finally:
        sys.path.pop(0)

    if not hasattr(journal_text_width, "TEXT_WIDTH"):
        raise RuntimeError(
            "journal_text_width.py exists but does not define TEXT_WIDTH.\n"
            "Example:\n"
            "    TEXT_WIDTH = 345.0"
        )

    try:
        return float(journal_text_width.TEXT_WIDTH)
    except Exception as e:
        raise RuntimeError(
            "journal_text_width.TEXT_WIDTH must be a number (float or int)."
        ) from e



def create_figure(textwidth_ratio, height_to_width_ratio=__DEFAULT_RATIO__, generator=plt.subplots, **kwargs):
    width = _load_text_width() * textwidth_ratio / 72. # 72 pts per inch 
    height = width * height_to_width_ratio
    return generator(figsize=(width, height), **kwargs)

def create_normal_figure(height_to_width_ratio=__DEFAULT_RATIO__, generator=plt.subplots, **kwargs):
    return create_figure(__NORMAL_TEXT_WIDTH_FRACTION__, height_to_width_ratio, generator, **kwargs)

def create_large_figure(height_to_width_ratio=.5 * __DEFAULT_RATIO__, generator=plt.subplots, **kwargs):
    return create_figure(__LARGE_TEXT_WIDTH_FRACTION__, height_to_width_ratio, generator, **kwargs)