"""
Drop-in stand-in for tqdm.tqdm without a progress bar, so the train/test/project loops keep
their original code (enumerate wrapped in tqdm, set_postfix_str per batch) without the
tqdm dependency or progress-bar spam in the slurm logs.
"""
class tqdm:
    def __init__(self, iterable, total=None, **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        return self.total

    def set_postfix_str(self, *args, **kwargs):
        pass
