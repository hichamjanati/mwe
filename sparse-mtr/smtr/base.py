"""Base class for all Estimators."""
import warnings
import numpy as np
from .utils import auc_prc, get_unsigned
from .exceptions import NotFittedError
from .ot import otfunctions


class BaseEstimator(object):
    """Base Class for all estimators.

    General class for multi-task regression problems of type:

    .. math::
        \min_{\theta^1, \dots} \frac{1}{2n}\sum_{k=1}\|X^k\theta^k - Y^k\|^2
        + \alpha g_1(\theta^1, \dots )

    Attributes
    ----------
    n_features : number of features
    n_samples : number of samples
    n_tasks : length of the lists `X`and `Y`.

    """
    def __init__(self, callback=False, n_jobs=1, classification=False,
                 **kwargs):
        """Construct instance.

        Parameters
        ----------
        alpha : float.
            Regularisation parameter.
        callback : boolean. optional.
            if True, set a callback function to the solver.

        """
        self.classification = classification
        self.callback = callback
        self.callback_kwargs = kwargs
        self._inspector = None
        self._solver = None
        self.n_jobs = n_jobs
        self.wyy0 = 0.

    def _pre_fit(self, X, Y):
        """Set data before fitting."""
        # if self.normalize:
        #     self.scaling_ = np.linalg.norm(X, axis=1)
        #     X = X / self.scaling_[:, None, :]
        #     self.callback_kwargs["rescaling"] = self.scaling_.T

        self._set_callback()

        return X, Y

    def _post_fit(self):
        """renormalize coefs if normalize is True."""
        # if self.normalize:
        #     self.coefs_ /= self.scaling_.T
        pass

    def _fit(self, X, Y, **kwargs):
        """Launch solver.

        Parameters
        ----------
        X : list of arrays of shape (n_samples, n_features).
            list of training arrays of shape (n_samples, n_features).
        Y : list of arrays of shape (n_samples,).
            list of target arrays of shape (n_samples,). Must have the same
            length as `X`.
        **kwargs: dict.
            supplementary parameters passed to the solver.

        Returns
        -------
        instance of self.

        """
        X, Y = self._pre_fit(X, Y)
        self.coefs_ = self._solver(self,
                                   callback=self.callback_f, **kwargs)
        self._post_fit()

        return self

    def _set_callback(self):
        """Set callback if `callback` is True."""
        self.callback_f = None
        if self.callback:
            self.callback_f = self._inspector(self.objective,
                                              **self.callback_kwargs)

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def predict(self, X):
        """Predict target Y given unseen data X.

        Returns list of y = x.dot(theta).

        Parameters
        ----------
        X: list of arrays of shape (None, n_features)

        Returns
        -------
        list of arrays of shape (None, n_samples)

        """
        if not hasattr(self, 'coefs_'):
            raise NotFittedError("Estimator not fitted !")
        zips = zip(X, self.coefs_.T)
        Y = np.array([x.dot(c) for x, c in zips])

        return Y

    def reset(self):
        if hasattr(self, 'coefs_'):
            del self.coefs_
        if hasattr(self, 'scaling_'):
            del self.scaling_

    def objective(self, coefs):
        pass

    def score_supports(self, coefs_true):
        """Test support recovery given the true coefs.

        Parameters
        ----------
        coefs_true: array, shape (n_features, n_tasks)
            true coefs.

        Returns
        -------
        boolean

        """
        if not hasattr(self, 'coefs_'):
            raise NotFittedError("""Estimator must be fitted before computing the
                                  support recovery score""")

        true_support = coefs_true > 0.
        test = (true_support == (self.coefs_ > 0.)).all()

        return test

    def quadraticloss(self, theta):
        r"""Compute the linreg loss function.

        Given a coefs matrix and the data matrices `X`[k] and `Y`[k]:

        .. math::

        1/2 * sum(\Vert Y_k - X_k.theta_k \Vert_2 )

        Parameters
        ----------
        theta : array, shape (n_features, n_tasks)
            regression coefs.

        Returns
        -------
        float.
            valued regression loss at theta.

        """
        trip = zip(self.Xs, self.Ys, theta.T)
        loss = [np.linalg.norm(y -
                               x.dot(t)) ** 2 for (x, y, t) in trip]
        return 0.5 * sum(loss)

    def ot_score(self, coefs_true, coefs_pred, M, epsilon, gamma=0., log=False,
                 wyy0=None, **kwargs):
        shape = coefs_true.shape
        n_tasks = shape[-1]
        coefs_hat = coefs_pred.reshape(shape)
        if epsilon:
            if (coefs_hat.reshape(-1, n_tasks).max(axis=0) < 1.).any():
                return 1e100
            f = otfunctions.ot_amari(coefs_true, coefs_hat,
                                     M,
                                     epsilon,
                                     gamma, log=log,
                                     wyy0=wyy0,
                                     normed=True,
                                     **kwargs)
            if np.isnan(f) and not log:
                print("Nan in OT metric, switched to log !")
                f = otfunctions.ot_amari(coefs_true, coefs_hat,
                                         M,
                                         epsilon,
                                         gamma, log=True,
                                         wyy0=wyy0,
                                         normed=True,
                                         **kwargs)
        else:
            f = otfunctions.emd(coefs_true, coefs_hat, M)

        return f

    def score(self, X, Y):
        if self.classification:
            ytrue = np.argmax(Y, axis=0)
            ypred = np.argmax(X[0].dot(self.coefs_), axis=1)
            mse = (ytrue != ypred).mean()
        else:
            Y_pred = self.predict(X)
            mse = ((Y_pred - Y) ** 2).mean(axis=1)
        return -mse

    def score_coefs(self, coefs_true,
                    precision=0, mean=True, M=None, epsilon=None,
                    compute_ot=True,
                    gamma=0., log=False, wyy0=None, **kwargs):
        """Compute metrics between fitted and true coefs.

        Parameters
        ----------
        coefs_true: array, shape (n_features, n_tasks)
            true coefs.
        precision: float (optional).
            between (0 - 1). support threshold for coefs_true.

        Returns
        -------
        dict ('auc, 'mse', 'ot').

        """
        if not hasattr(self, 'coefs_'):
            raise NotFittedError("""Estimator must be fitted before computing the
                                  estimation score""")

        if np.isnan(self.coefs_).any():
            self.reset()
            warnings.warn("AAAAAAAAAAAAAAAAAAAAAAA NAN COEFS")
            return dict(auc=100, mse=-1000, ot=-1000, aucabs=100, otabs=-1000)
        true_parts = get_unsigned(coefs_true)
        pred_parts = get_unsigned(self.coefs_)
        auc, mse, ot = 0., 0., 0.
        i = 0
        if wyy0:
            w = 0.
        else:
            w = None
        for true, pred in zip(true_parts, pred_parts):
            if not true.any():
                continue
            i += 1
            auc += auc_prc(true, pred, precision, mean)
            coefs_hat = pred.copy().flatten()
            true_ = true.flatten()
            mse_ = ((true_ - coefs_hat) ** 2).mean()
            mse_ /= max(true_.max(), coefs_hat.max(), 1)
            mse += mse_
            if compute_ot:
                ot += self.ot_score(true, pred, M=M, epsilon=epsilon,
                                    gamma=gamma, log=log, wyy0=w,
                                    **kwargs)
            else:
                ot = 1e100
        if wyy0 and compute_ot:
            ot -= wyy0 / 2

        aucabs = auc_prc(abs(coefs_true), abs(self.coefs_))
        otabs = 1e100
        if compute_ot:
            otabs = self.ot_score(abs(coefs_true), abs(self.coefs_), M,
                                  epsilon=epsilon, gamma=gamma, log=log,
                                  wyy0=w, **kwargs)
        i = max(i, 1)
        metrics = dict(auc=auc / i, mse=-mse / i, ot=-ot / i, aucabs=aucabs,
                       otabs=-otabs)
        return metrics
