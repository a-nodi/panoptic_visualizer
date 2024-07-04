def unzip(enumerable):
    return [list(tupled_pair) for tupled_pair in list(zip(*enumerable))]