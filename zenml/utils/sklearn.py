from sklearn.preprocessing import FunctionTransformer
import inspect


def pipelinize(function, *args):
    """
    Take a custom function and make it compatible with sklearn's pipeline
    """

    keys = inspect.getfullargspec(function).args[1:]
    values = list(args)
    return FunctionTransformer(function, validate=False, kw_args=dict(zip(keys, values)))

