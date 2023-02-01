import math
def sigmoid(x):
    ax_b = (50*x) - 43.5
    return 1 / (1 + math.exp(-1*ax_b))