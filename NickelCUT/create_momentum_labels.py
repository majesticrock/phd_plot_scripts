import numpy as np

def create_momentum_labels(L):
    """Create tick positions and labels for momentum space plots.
    Labels follow: pi * (2*i/L - 1), simplified as much as possible."""
    from math import gcd
    
    ticks = np.arange(L)
    labels = []
    for i in range(L):
        numerator = 2 * i - L
        denominator = L
        
        # Simplify the fraction
        g = gcd(abs(numerator), denominator)
        numerator //= g
        denominator //= g
        
        # Format the label
        if numerator == 0:
            label = r"$0$"
        elif denominator == 1:
            if numerator == 1:
                label = r"$\pi$"
            elif numerator == -1:
                label = r"$-\pi$"
            else:
                label = rf"${numerator}\pi$"
        else:
            if numerator == 1:
                label = rf"$\frac{{\pi}}{{{denominator}}}$"
            elif numerator == -1:
                label = rf"$-\frac{{\pi}}{{{denominator}}}$"
            else:
                label = rf"$\frac{{{numerator}\pi}}{{{denominator}}}$"
        
        labels.append(label)
    
    return ticks, labels