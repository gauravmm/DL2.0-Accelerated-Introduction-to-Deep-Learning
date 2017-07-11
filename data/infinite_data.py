# Convert data into a stream of never-ending data
def wrap(data, batch_size):
    x, y = data
    i = 0 # Type: int
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= x.shape[0]:
            rv = list(range(i, x.shape[0])) + list(range(0, j - x.shape[0]))
            yield (x[rv,...], y[rv])
            i = j - x.shape[0]
        else:
            yield (x[i:j,...], y[i:j])
            i = j
