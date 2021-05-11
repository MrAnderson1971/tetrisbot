def clamp(n, min, max):
    '''clamps a number between min and max'''

    if n < min:
        return min
    if n > max:
        return max
    return n
