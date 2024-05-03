def get_dataset(dataset, **kwargs):
    if dataset == 'nordland':
        from .nordland import Nordland
        return Nordland(**kwargs)
    if dataset == 'tokyo':
        from .tokyo import Tokyo
        return Tokyo(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
