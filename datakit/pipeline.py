def pipeline_load(pipeline, operators):
    """
    load a pipeline from a list of dicts

    Parameters
    ----------
    pipeline : list of dicts of structure {'name': str, 'params': dict}
        the list of operations to apply where the operatorions behavior
        are defined in operators
    operators : dict of str->callable
        each operator transforms an iterator to another iterator
    """
    iterator = None
    for op in pipeline:
        name, params = op['name'], op['params']
        iterator = operators[name](iterator, **params)
    return iterator
