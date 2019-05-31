"""Multitask custom Exceptions."""


class NotFittedError(AttributeError):
    """Raised if an estimator is used before fitting."""


class LowWassersteinReg(RuntimeWarning):
    """Raised if Sinkhorn algorithm overflows / underflows."""


class LowWassersteinMaxiter(RuntimeWarning):
    """Raised if Sinkhorn algorithm does not converge."""


class NumericalInstability(RuntimeWarning):
    """Raised if the solver diverges."""
