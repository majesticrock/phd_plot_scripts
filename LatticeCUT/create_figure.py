import matplotlib.pyplot as plt

__TEXT_WIDTH__ = 390
__GOLDEN_RATIO__ = (5.**.5 - 1.) / 2.
__DEFAULT_RATIO__ = 4.8 / 6.4

__NORMAL_TEXT_WIDTH_FRACTION__ = 0.5
__LARGE_TEXT_WIDTH_FRACTION__ = 0.95

def create_figure(textwidth_ratio, height_to_width_ratio=__DEFAULT_RATIO__, generator=plt.subplots, **kwargs):
    width = __TEXT_WIDTH__ * textwidth_ratio / 72. # 72 pts per inch 
    height = width * height_to_width_ratio
    return generator(figsize=(width, height), **kwargs)

def create_normal_figure(height_to_width_ratio=__DEFAULT_RATIO__, generator=plt.subplots, **kwargs):
    return create_figure(__NORMAL_TEXT_WIDTH_FRACTION__, height_to_width_ratio, generator, **kwargs)

def create_large_figure(height_to_width_ratio=.5 * __DEFAULT_RATIO__, generator=plt.subplots, **kwargs):
    return create_figure(__LARGE_TEXT_WIDTH_FRACTION__, height_to_width_ratio, generator, **kwargs)