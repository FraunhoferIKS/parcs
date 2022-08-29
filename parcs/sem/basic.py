from dtaidistance import dtw
import numpy as np
import pandas as pd
from scipy.special import expit
from parcs.simulators.temporal.deterministic import FourierSeries


class FrequencyLogNormalLatents:
    """
    Sample Frequencies from log normal distribution
    by forcing less dominant frequencies to increase by a multiplicative factor

    Parameters
    ----------
    num_freqs : int
        number of frequencies to sample
    first_freq_mean : float
        The log normal mean value for the first (dominant) frequency
    next_freq_ratio : float
        The multiplicative factor, based on which the next frequencies increase
    sigma : float
        the scale of the log normal distribution
    frequency_prefix : str, default='w'
        the prefix of the data column names

    Attributes
    ----------
    self.data : pd.DataFrame
        the simulated data after calling the .sample() method

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> fln = FrequencyLogNormalLatents(
    ...     num_freqs=3,
    ...     first_freq_mean=np.pi/5,
    ...     next_freq_ratio=2.5,
    ...     sigma=np.pi/5
    ... )
    >>> np.round(fln.sample(sample_size=5), 2)
        w_0    w_1     w_2
    0  5.20   1.13  127.19
    1  1.28  14.40   13.91
    2  1.35   2.98   41.45
    3  0.96   5.88   39.87
    4  3.23   4.11  103.48
    """
    def __init__(self,
                 num_freqs: int = None,
                 first_freq_mean: float = None,
                 next_freq_ratio: float = None,
                 sigma: float = None,
                 frequency_prefix: str = 'str'):
        self.data = None
        self.num_freqs = num_freqs
        self.first_freq_mean = first_freq_mean
        self.next_freq_ratio = next_freq_ratio
        self.sigma = sigma

    def sample(self, sample_size: int = 100):
        """
        sample the latents

        Parameters
        ----------
        sample_size : int, default=100
            number of samples

        Returns
        -------
        self.data : pd.DataFrame
            the sampled values in a dataframe

        """
        means_ = [
            np.log(self.first_freq_mean) + i * np.log(self.next_freq_ratio)
            for i in range(self.num_freqs)
        ]
        self.data = pd.DataFrame({
            'w_{}'.format(i): np.random.lognormal(means_[i], self.sigma, size=sample_size)
            for i in range(self.num_freqs)
        })
        return self.data


class IndependentNormalLatents:
    """
    simulate independent normally distributed latent variables

    refer to user guide

    Examples
    --------
    >>> import numpy as np
    >>> from parcs.sem.basic import IndependentNormalLatents
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
        normal_vars = {
            var['name']: np.random.normal(var['mean'], var['sigma'], size=sample_size)
            for var in self.var_list if not var['log']
        }
        log_normal_vars = {
            var['name']: np.random.lognormal(np.log(var['mean']), var['sigma'], size=sample_size)
            for var in self.var_list if var['log']
        }
        self.data = pd.DataFrame({**normal_vars, **log_normal_vars})
        return self.data


class IndependentUniformLatents:
    """
        simulate independent uniformly distributed latent variables

        refer to user guide

        Examples
        --------
        >>> import numpy as np
        >>> from parcs.sem.basic import IndependentUniformLatents
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
    >>> from parcs.sem.basic import LatentLabelMaker
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


class ShapeletPlacementLabelMaker:
    """
    make label based on palcing a shapelet for a class of signals

    Parameters
    ----------
    window_ratio: float, defaul=0.3
        shapelet to sequence length ratio
    class_ratio: float, default=0.5
        class 1 to 0 ratio
    shapelet_num_sin: int, default=10
        number of sins in the fourier series for creating a random shapelet
    shapelet_added_noise: float, default=0.1
        added noise for Fourier Series of the shapelet

    Attributes
    ----------
    self.shapelet: array-like
        the generated shapelet
    self.shapelet_placement_idx: array-like
        array of length n, the sample size. For each sample, it is -1 if the sample
        doesn't have the shapelet. and it is a position index, the starting index of shapelet in
        the signal, if the signal has a shapelet

    Examples
    --------
    >>> from parcs.simulators.temporal.stochastic_processes import BrownianMotion
    >>> from parcs.sem.basic import IndependentUniformLatents
    >>> from parcs.sem.basic import ShapeletPlacementLabelMaker
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> # define the base time series
    >>> seq_len = 100
    >>> inl = IndependentUniformLatents().set_nodes(var_list=[{'name': 'drift', 'low': -1, 'high': 1}])
    >>> l = inl.sample(sample_size=10)
    >>> l['scale'] = 1.7
    >>> l['init'] = 0
    >>> bm = BrownianMotion(sampled_latents=l, geometric=False)
    >>> signals = bm.sample(seq_len=seq_len)
    >>> # make labels
    >>> lm = ShapeletPlacementLabelMaker()
    >>> labels = lm.make_label(signals=signals)
    >>> labels
    array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    >>> lm.shapelet_placement_idx
    array([47, 38, -1, 49, -1, 41, -1, 39, -1, -1])
    """
    def __init__(self,
                 window_ratio: float = 0.3,
                 class_ratio: float = 0.5,
                 shapelet_num_sin: int = 10,
                 shapelet_added_noise: float = 0.1):
        self.window_ratio = window_ratio
        self.is_shapelet_defined = False
        self.class_ratio = class_ratio
        self.shapelet = None
        self.shapelet_num_sin = shapelet_num_sin
        self.shapelet_added_noise = shapelet_added_noise

        self.seq_len = None
        self.shap_len = None

        self.shapelet_placement_idx = None

    def _make_shapelet(self):
        assert self.shap_len >= 5

        f_range = [np.pi / self.shap_len, np.pi / 4]
        p_range = [-np.pi / 4, np.pi / 4]
        freqs = np.random.uniform(f_range[0], f_range[1], size=self.shapelet_num_sin)
        phis = np.random.uniform(p_range[0], p_range[1], size=self.shapelet_num_sin)
        l = np.concatenate([freqs, phis]).reshape(1, -1)
        l = pd.DataFrame(
            l,
            columns=['w_{}'.format(i) for i in range(self.shapelet_num_sin)] +
                    ['phi_{}'.format(i) for i in range(self.shapelet_num_sin)]
        )
        fs = FourierSeries(
            sampled_latents=l,
            amplitude_exp_decay_rate=0,
            added_noise_sigma_ratio=self.shapelet_added_noise
        )
        self.shapelet = fs.sample(seq_len=self.shap_len)[0]
        self.is_shapelet_defined = True

    def _place_shapelet(self, signal: np.array = None):
        assert self.is_shapelet_defined
        # choose a random position skipping 2 values from start and end
        pos = np.random.choice(np.arange(2, self.seq_len-self.shap_len))
        shap_offset = signal[pos]
        shap = self.shapelet + (shap_offset - self.shapelet[0])
        sig_offset = shap[-1]
        signal[pos:pos + self.shap_len] = shap
        signal[pos + self.shap_len:] += (sig_offset - signal[pos + self.shap_len])
        return signal, pos

    def make_label(self, signals: np.array = None):
        """
        make labels for signals

        Parameters
        ----------
        signals: array-like
            n x t array of signals to be labeled

        Returns
        -------
        signals: array-like
            same shape as input signals, but with shapelets placed for class 1
        labels: array-like
            array of length n the labels for samples

        """
        n, self.seq_len = signals.shape
        self.shapelet_placement_idx = -np.ones(shape=(n,)).astype(int)
        self.shap_len = int(self.seq_len*self.window_ratio)
        if not self.is_shapelet_defined:
            self._make_shapelet()
        signal_inds = np.random.choice(range(n), size=int(n*self.class_ratio), replace=False)
        for idx in signal_inds:
            signals[idx], shapelet_pos = self._place_shapelet(signal=signals[idx])
            self.shapelet_placement_idx[idx] = shapelet_pos

        labels = np.zeros(shape=(n,)).astype(int)
        labels[signal_inds] = 1
        return signals, labels


class DetShapeletPlacementLabelMaker:
    """
    make label based on placing a shapelet for a class of signals at a determined index

    Parameters
    ----------
    shapelet_length: int,
        length of the shapelet
    shapelet_index: int,
        index of the first point of shapelet in the signal
    class_ratio: float, default=0.5
        class 1 to 0 ratio
    shapelet_num_sin: int, default=10
        number of sins in the fourier series for creating a random shapelet
    shapelet_added_noise: float, default=0.1
        added noise for Fourier Series of the shapelet

    Attributes
    ----------
    self.shapelet: array-like
        the generated shapelet

    Examples
    --------
    >>> from parcs.simulators.temporal.stochastic_processes import BrownianMotion
    >>> from parcs.sem.basic import IndependentUniformLatents
    >>> from parcs.sem.basic import DetShapeletPlacementLabelMaker
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> # define the base time series
    >>> seq_len = 100
    >>> inl = IndependentUniformLatents().set_nodes(var_list=[{'name': 'drift', 'low': -1, 'high': 1}])
    >>> l = inl.sample(sample_size=10)
    >>> l['scale'] = 1.7
    >>> l['init'] = 0
    >>> bm = BrownianMotion(sampled_latents=l, geometric=False)
    >>> signals = bm.sample(seq_len=seq_len)
    >>> # make labels
    >>> lm = DetShapeletPlacementLabelMaker(shapelet_length=10, shapelet_index=4)
    >>> signals, labels = lm.make_label(signals=signals)
    >>> labels
    array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    """
    def __init__(self,
                 shapelet_length: int = None,
                 shapelet_index: int = None,
                 class_ratio: float = 0.5,
                 shapelet_num_sin: int = 10,
                 shapelet_added_noise: float = 0.1):

        self.is_shapelet_defined = False
        self.class_ratio = class_ratio
        self.shapelet = None
        self.shapelet_num_sin = shapelet_num_sin
        self.shapelet_added_noise = shapelet_added_noise

        self.seq_len = None
        self.shap_len = shapelet_length
        self.shap_idx = shapelet_index

    def _make_shapelet(self):
        f_range = [np.pi / self.shap_len, np.pi / 4]
        p_range = [-np.pi / 4, np.pi / 4]
        freqs = np.random.uniform(f_range[0], f_range[1], size=self.shapelet_num_sin)
        phis = np.random.uniform(p_range[0], p_range[1], size=self.shapelet_num_sin)
        l = np.concatenate([freqs, phis]).reshape(1, -1)
        l = pd.DataFrame(
            l,
            columns=['w_{}'.format(i) for i in range(self.shapelet_num_sin)] +
                    ['phi_{}'.format(i) for i in range(self.shapelet_num_sin)]
        )
        fs = FourierSeries(
            sampled_latents=l,
            amplitude_exp_decay_rate=0,
            added_noise_sigma_ratio=self.shapelet_added_noise
        )
        self.shapelet = fs.sample(seq_len=self.shap_len)[0]
        self.is_shapelet_defined = True

    def _place_shapelet(self, signal: np.array = None):
        assert self.is_shapelet_defined
        pos = self.shap_idx
        shap_offset = signal[pos]
        shap = self.shapelet + (shap_offset - self.shapelet[0])
        sig_offset = shap[-1]
        signal[pos:pos + self.shap_len] = shap
        signal[pos + self.shap_len:] += (sig_offset - signal[pos + self.shap_len])
        return signal, pos

    def make_label(self, signals: np.array = None):
        """
        make labels for signals

        Parameters
        ----------
        signals: array-like
            n x t array of signals to be labeled

        Returns
        -------
        signals: array-like
            same shape as input signals, but with shapelets placed for class 1
        labels: array-like
            array of length n the labels for samples

        """
        n, self.seq_len = signals.shape
        if not self.is_shapelet_defined:
            self._make_shapelet()
        signal_inds = np.random.choice(range(n), size=int(n*self.class_ratio), replace=False)
        for idx in signal_inds:
            signals[idx], shapelet_pos = self._place_shapelet(signal=signals[idx])

        labels = np.zeros(shape=(n,)).astype(int)
        labels[signal_inds] = 1
        return signals, labels


if __name__ == '__main__':
    o = FrequencyLogNormalLatents(first_freq_mean=1, next_freq_ratio=2)
    o.sample()