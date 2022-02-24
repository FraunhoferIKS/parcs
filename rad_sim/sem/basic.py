import matplotlib.pyplot as plt
from dtaidistance import dtw
import numpy as np
import pandas as pd
from scipy.special import expit


class IndependentNormalLatents:
    """
    simulate independent normally distributed latent variables

    refer to user guide

    Examples
    --------
    >>> import numpy as np
    >>> from rad_sim.sem.basic import IndependentNormalLatents
    >>> np.random.seed(1)
    >>> inl = IndependentNormalLatents()
    >>> inl.set_nodes(var_list=[{'name': 'x', 'mean': 1, 'sigma': 2}, {'name': 'y', 'mean': -2, 'sigma': 1}])
    >>> np.round(inl.sample(sample_size=4), 2)
          x     y
    0  4.25 -1.13
    1 -0.22 -4.30
    2 -0.06 -0.26
    3 -1.15 -2.76
    """
    def __init__(self):
        self.data = None
        self.var_list = None

    def set_nodes(self, var_list=None):
        """
        set the list of variables to be simulated

        Parameters
        ----------
        var_list : list of dicts
            in the form of:
            `[{'name': 'x', 'mean': 1, 'sigma':1}, ...]`

        Returns
        -------
        self
        """
        self.var_list = var_list
        return self

    def sample(self, sample_size: int = 100):
        """

        Parameters
        ----------
        sample_size : int, default=100
            number of realizations

        Returns
        -------
        df : pd.DataFrame
            `(n,p)` realizations for n samples and p latent variables
        """
        self.data = pd.DataFrame({
            var['name']: np.random.normal(var['mean'], var['sigma'], size=sample_size)
            for var in self.var_list
        })
        return self.data


class IndependentUniformLatents:
    """
        simulate independent uniformly distributed latent variables

        refer to user guide

        Examples
        --------
        >>> import numpy as np
        >>> from rad_sim.sem.basic import IndependentUniformLatents
        >>> np.random.seed(1)
        >>> inl = IndependentUniformLatents()
        >>> inl.set_nodes(var_list=[{'name': 'x', 'low': 1, 'high': 2}, {'name': 'y', 'low': 50, 'high': 60}])
        >>> np.round(inl.sample(sample_size=4), 2)
              x      y
        0  1.42  51.47
        1  1.72  50.92
        2  1.00  51.86
        3  1.30  53.46
        """
    def __init__(self):
        self.data = None
        self.var_list = None

    def set_nodes(self, var_list=None):
        """
        set the list of variables to be simulated

        Parameters
        ----------
        var_list : list of dicts
            in the form of:
            `[{'name': 'x', 'low': 1, 'high':1}, ...]`

        Returns
        -------
        self
        """
        self.var_list = var_list
        return self

    def sample(self, sample_size: int = 100):
        """
        Parameters
        ----------
        sample_size : int, default=100
            number of realizations

        Returns
        -------
        df : pd.DataFrame
            `(n,p)` realizations for n samples and p latent variables
        """
        self.data = pd.DataFrame({
            var['name']: np.random.uniform(var['low'], var['high'], size=sample_size)
            for var in self.var_list
        })
        return self.data


class LatentLabelMaker:
    """
    Making label based on sampled labels

    in user guide

    Parameters
    ----------
    coef_min : float, default=1.0
        min value in uniform distribution
    coef_max : float, default=4.0
        max value in uniform distribution
    normalize_latent : bool, default=True
        if latent variables must be normalized before multiplication to coef vector
    sigmoid_offset : float, default=0
        it affects the mean of label. if offset `>0`, then mean is closer to 1

    Attributes
    ----------
    self.coefs : array-like
        the sampled coef vector for to be multiplied to latents

    Examples
    --------
    >>> import numpy as np
    >>> from rad_sim.sem.basic import LatentLabelMaker
    >>> np.random.seed(1)
    >>> llm = LatentLabelMaker(coef_min=0, coef_max=4, sigmoid_offset=1)
    >>> latents = np.random.normal(0, 1, size=(10, 2))
    >>> llm.make_label(sampled_latents=latents)
    array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1])
    >>> np.round(llm.coefs, 3)
    array([2.768, 1.262])
    """
    def __init__(self,
                 coef_min: float = 1,
                 coef_max: float = 4,
                 normalize_latent: bool = True,
                 sigmoid_offset: float = 0):
        self.normalize_latent = normalize_latent
        self.coef_min = coef_min
        self.coef_max = coef_max
        self.offset = sigmoid_offset

        self.coefs_sampled = False
        self.coefs = None

    def _sample_coefs(self, size=None):
        self.coefs = np.random.uniform(self.coef_min, self.coef_max, size=size)
        self.coefs_sampled = True
        return self

    @staticmethod
    def _normalize(x):
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def make_label(self, sampled_latents: np.array = None):
        """
        make y labels based on latents

        user guide

        Parameters
        ----------
        sampled_latents : array-like
            `n x p` for n samples and p number of latents

        Returns
        -------
        y : array-like
            n binary labels for n samples
        """
        if self.normalize_latent:
            sampled_latents = self._normalize(sampled_latents)
        if not self.coefs_sampled:
            self._sample_coefs(size=sampled_latents.shape[1])
        states = np.dot(sampled_latents, self.coefs)
        norm_states = self.offset + self._normalize(states)
        probs = expit(norm_states)
        return np.array([
            np.random.choice([0, 1], p=[1-prob, prob])
            for prob in probs
        ])


class ShapeletDTWLabelMaker:
    """
    label maker based on shapelets in series

    in user guide

    Parameters
    ----------
    window_ratio : float, default=0.1
        shapelet-to-sequence length ratio
    fast_approximation : bool, default=False
        if True, a number of `sub_population` samples are sampled randomly from the signals
        to find the reference shapelet. for `n_sample > 1000` it's recommended to do so.
        if set to False, all the signals will be used to find the reference shapelet
    parallel : bool, default=True
        if set to `True`, parallel processing will be used in distance calculation
    sub_population : int, default=700
        if `fast_approximation == True`, this will be the number of signals used to find the reference shapelet

    Attributes
    ----------
    self.ref_shapelet : array-like
        the reference shapelet that has been set to label 0
    self.success_probs : array-like
        the success probabilities for each sample, to be used in Bernoulli sampling
    """
    def __init__(self,
                 window_ratio: float = 0.1,
                 fast_approximation: bool = False,
                 parallel: bool = True,
                 sub_population: int = 700):
        self.window_ratio = window_ratio
        self.window_len = None
        self.distance_func = dtw.distance_matrix_fast
        self.parallel = parallel
        self.fast_approx = fast_approximation
        self.sub_population = sub_population

        self.ref_shapelet = None
        self.success_probs = None
        self.is_ref_set = False

    def _make_ref(self, shapelets):
        if self.fast_approx:
            shap_dict = shapelets[
                np.random.choice(np.arange(shapelets.shape[0]), replace=False, size=self.sub_population)
            ]
        else:
            shap_dict = shapelets
        dtw_dist = self.distance_func(shap_dict)
        ref_shapelet_idx = np.argmin(np.median(dtw_dist, axis=0))
        self.ref_shapelet = shap_dict[ref_shapelet_idx]
        self.is_ref_set = True
        return dtw_dist[ref_shapelet_idx]

    def make_label(self, signals: np.array = None):
        """
        make label

        user guide

        Parameters
        ----------
        signals : array-like
            `n x t` n signals with t sequence length, to be labeled

        Returns
        -------
        y : array-like
            the produced labels
        """
        if not self.is_ref_set:
            self.window_len = int(signals.shape[1] * self.window_ratio)
            score_row = self._make_ref(shapelets=signals[:, -self.window_len:])
        else:
            score_row = np.array([
                dtw.distance_fast(self.ref_shapelet, signals[i, -self.window_len:], {'parallel': self.parallel})
                for i in range(signals.shape[0])
            ])

        score_row = (score_row - score_row.mean()) / score_row.std()
        self.success_probs = expit(score_row)
        return np.array([
            np.random.choice([0, 1], p=[1 - prob, prob])
            for prob in self.success_probs
        ])


if __name__ == '__main__':
    def func(x):
        y = x.copy()
        y = y + np.random.normal(0, 1)
        print(x)
        print(y)

    x = np.array([1, 2, 3])
    func(x)
    print(x)