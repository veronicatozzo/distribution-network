
from itertools import chain, combinations
import numbers
import warnings
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from scipy import sparse

from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES, _deprecate_positional_args)

from sklearn.preprocessing import StandardScaler


class SetStandardScaler(StandardScaler):
    """Standardize sample sets by centering the distribution in zero and scaling them to unit variance
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the sets in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.
    
    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    Attributes
    ----------
    scale_ : ndarray or None, shape (n_features,)
        Per feature relative scaling of the data. This is calculated using
        `np.sqrt(var_)`. Equal to ``None`` when ``with_std=False``.
        .. versionadded:: 0.17
           *scale_*
    mean_ : ndarray or None, shape (n_features,)
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.
    var_ : ndarray or None, shape (n_features,)
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.
    n_samples_seen_ : int or array, shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are not missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.
    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = [S1, S2, S3, ...., SN]
    >>> scaler = SetStandardScaler()
    >>> scaler.fit(data)
    >>> scaler.transform(data)

    """ 

    def __init__(self, copy=True, with_mean=True, with_std=True, stream=True):
        self.copy=copy
        self.with_mean=with_mean
        self.with_std=with_std
        self.stream=stream

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()

        if self.stream and not isinstance(X[0], str):
            raise ValueError('In the case of stream a list of path to arrays has to be passed')
            

        if self.stream:
            stds = []
            means = []
            for f in X:
                try:
                    x = np.load(f)
                    means.append(np.mean(x, axis=0))
                    stds.append(np.std(x, axis=0))
                except:
                    continue
            self.mean_ = np.mean(means, axis=0)
            self.scale_ = np.mean(stds, axis=0)
            self.n_features_in = self.mean_.shape[0]
        else:
            X = [self._validate_data(x, accept_sparse=('csr', 'csc'),
                                    estimator=self, dtype=FLOAT_DTYPES,
                                    force_all_finite='allow-nan') for x in X]

            if self.with_mean:
                self.mean_ = np.mean(np.vstack(X), axis=0)
            self.scale_ = np.std(np.vstack(X), axis=0)
            
        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
       # check_is_fitted(self)

        # copy = copy if copy is not None else self.copy
        # X = [self._validate_data(x, reset=False,
        #                         accept_sparse='csr', copy=copy,
        #                         estimator=self, dtype=FLOAT_DTYPES,
        #                         force_all_finite='allow-nan')
        #      for x in X]

        if self.with_mean:
            X = [x -self.mean_ for x in X]
        if self.with_std:
            X = [x/self.scale_ for x in X]
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = np.asarray(X)
        if copy:
            X = X.copy()
        if self.with_std:
            X =  [x*self.scale_ for x in X]
        if self.with_mean:
            X  = [x+self.mean_ for x in X]
        return X
