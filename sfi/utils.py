import itertools


def batched(iterable, n):
    counter = itertools.count()

    for _, group in itertools.groupby(iterable, lambda _: next(counter) // n):
        yield list(group)


def files(path):
    return sorted([p for p in path.iterdir() if p.is_file()])


# Proper weight decay for Adam, not L2 penalty
# https://github.com/pytorch/pytorch/pull/4429
def decay_weights(optimizer, v):
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.data.add_(-v * group["lr"])
