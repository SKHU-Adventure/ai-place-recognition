def get_dataset(dataset, **kwargs):
    if dataset == 'nordland':
        from .nordland import Nordland
        return Nordland(**kwargs)
    if dataset == 'tokyo':
        from .tokyo import Tokyo
        return Tokyo(**kwargs)
    if dataset == 'skhu1':
        from .skhu1 import SKHU1
        return SKHU1(**kwargs)
    if dataset == 'skhu2':
        from .skhu2 import SKHU2
        return SKHU2(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
