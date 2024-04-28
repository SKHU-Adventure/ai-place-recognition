def get_dataset(dataset, **kwargs):
    if dataset == 'nordland':
        from .nordland import Nordland
        return Nordland(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
