import numpy as np
import pandas as pd
from modules.utils import *
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import PolynomialFeatures


class EdgeHandler:
    def __init__(self,
                 complexity=None,
                 adj_matrix=None):
        # set the noise
        # limit options
        self.complexity = complexity
        self.edge_func_options = {
            'identity': self.__identity,
            'linear': self.__linear,
            'sigmoid': self.__sigmoid,
            'sinusoidal': self.__sinusoid
        }

        self.edge_functions = None
        self._pick_edge_functions(adj_matrix=adj_matrix)


    @staticmethod
    def __identity(array=None):
        return array

    @staticmethod
    def __linear(array=None):
        coef = np.random.normal(loc=1, scale=3)
        return array * coef

    @staticmethod
    def __sigmoid(array: np.ndarray = None):
        # 1/1+exp(k(c-x))
        # k in [-10, -1], [1, 10]
        # c a threshold in the data
        c = get_random_cut(array=array)
        k = np.random.choice([-1, 1]) * np.random.uniform(low=1, high=10)
        return [
            1/(1 + np.exp(k*(c-i))) for i in array
        ]

    @staticmethod
    def __sinusoid(array: np.ndarray=None):
        # a.sin(bx+c)
        # a in [-3, -1], [1, 3]
        # b in [1.6, 14]
        # c in [0, 2pi]
        scaled_array = minmax_scale(array)
        a = np.random.choice([-1, 1]) * np.random.uniform(low=1, high=3)
        b = np.random.uniform(low=1.5, high=14)
        c = np.random.uniform(low=0, high=2*np.pi)
        return a * np.sin(b * scaled_array + c)

    def _pick_edge_functions(self, adj_matrix=None):
        print('adj matrix:', adj_matrix)
        num_nodes = adj_matrix.shape[0]
        print('num nodes:', num_nodes)
        self.edge_functions = np.random.choice(
            self.get_options(),
            size=(num_nodes, num_nodes),
            p=self.get_probs()
        )
        self.edge_functions = mask_matrix(
            matrix=self.edge_functions,
            mask=adj_matrix,
            mask_value='null'
        )
        return None

    def generate_input_values(self,
                              data=None,
                              parents=None,
                              node_number=None):
        inp_data = {p: None for p in parents}
        for p in parents:
            func = self.edge_functions[int(p[1:]), node_number]
            inp_data[p] = self.edge_func_options[func](array=data[p])

        return pd.DataFrame(inp_data)

    def get_options(self):
        return (
            'identity', 'linear',
            'sigmoid', 'sinusoidal'
        )

    def get_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_options())
        )


class StateHandler:
    def __init__(self,
                 num_nodes=None,
                 complexity=None):
        # TODO: implement noise
        self.complexity = complexity
        self.exogenous_dists = {
            'uniform': {
                'function': np.random.uniform,
                'args': {
                    'low': 0,
                    'high': 1
                }
            },
            'normal': {
                'function': np.random.normal,
                'args': {
                    'loc': 0,
                    'scale': 1
                }
            }
        }
        self.state_func_options = {
            'linear': self.__linear,
            'poly2': self.__poly2,
            'poly3': self.__poly3
        }

        self._pick_state_functions(num_nodes=num_nodes)

    @staticmethod
    def __linear(data: pd.DataFrame=None):
        num_features = len(data.columns)

        return np.dot(
            data.values,
            np.random.uniform(low=-10, high=10, size=(num_features, 1))
        )

    @staticmethod
    def __poly_n(n=None):
        return PolynomialFeatures(
            degree=n,
            interaction_only=np.random.choice([True, False]),
            include_bias=np.random.choice([True, False])
        )

    def __poly2(self, data: pd.DataFrame = None):

        transformed_data = self.__poly_n(2).fit_transform(data)
        return self.__linear(data=pd.DataFrame(transformed_data))

    def __poly3(self, data: pd.DataFrame = None):

        transformed_data = self.__poly_n(3).fit_transform(data)
        return self.__linear(data=pd.DataFrame(transformed_data))

    def _pick_state_functions(self, num_nodes=None):
        self.state_funcs = np.random.choice(
            self.get_options(),
            size=num_nodes,
            p=self.get_probs()
        )

    def _generate_no_parent_state_value(self,
                                        size=None):
        # TODO: handle the selected distribution with complexity
        distribution = self.exogenous_dists[
            np.random.choice(
                self.get_exogenous_dist_options()
            )
        ]
        return distribution['function'](
            size=(size,),
            **distribution['args']
        )

    def generate_state_value(self,
                             inputs=None,
                             dataset_size=None,
                             node_number=None):
        if len(inputs.columns) != 0:
            return self.state_func_options[
                self.state_funcs[node_number]
            ](data=inputs)
        else:
            return self._generate_no_parent_state_value(size=dataset_size)

    def generate_zrx_state_value(self,
                                 inputs=None,
                                 dataset_size=None,
                                 node_number=None):
        # feed z parents in linear/poly2/poly3
        # feed x individually
        ##      for observed parts: do something
        ##      for missing part: add nothing
        # feed R individually
        ##      dichotomous probabilities for 1-0
        ##      implement attrition/exclusive relations
        ## Log-add the evidences
        # return R probs
        return [1]


    def get_options(self):
        return (
            'linear', 'poly2', 'poly3'
        )

    def get_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_options())
        )

    def get_exogenous_dist_options(self):
        return (
            'normal', 'uniform'
        )


class OutputHandler:
    def __init__(self,
                 num_nodes=None,
                 complexity=None):
        # state the noise
        # limit the options
        self.complexity = complexity
        self.measurement_func_options = {
            'identity': self.__identity,
            'binary_threshold': self.__binary_threshold,
            'binary_random_draw': self.__binary_random_draw

        }
        self.output_func_options = {
            'identity': self.__identity,
            'scale': self.__scale
        }

        self.output_funcs = None
        self._pick_output_functions(num_nodes=num_nodes)

    @staticmethod
    def __identity(array):
        return array

    @staticmethod
    def __binary_threshold(array: np.ndarray = None):
        threshold = get_random_cut(array=array)
        return [
            0 if i <= threshold else 1 for i in array
        ]

    @staticmethod
    def __binary_random_draw(array: np.ndarray=None):
        scaled_array = minmax_scale(array)
        try:
            assert scaled_array.shape[1] != 1
        except AssertionError:
            scaled_array = scaled_array.transpose()[0]
        finally:
            return [
                np.random.choice(
                    [0, 1],
                    p=[1 - i, i]
                ) for i in scaled_array
            ]

    @staticmethod
    def __scale(array: np.ndarray=None):
        coef = np.random.uniform(low=1, high=3) * np.random.choice([-1, 1])
        return minmax_scale(array)*coef

    def _pick_output_functions(self, num_nodes=None):
        self.output_funcs = np.random.choice(
            self.get_options(),
            size=num_nodes,
            p=self.get_probs()
        )

    def generate_output_value(self,
                              array=None,
                              node_number=None):
        output_function = self.output_funcs[node_number]
        return self.output_func_options[output_function](array=array)

    def generate_measurement_dataset(self,
                                     data: pd.DataFrame = None):
        measured_data = data.copy()
        # based on the output value
        for c in measured_data.columns:
            measurement_func = np.random.choice(
                ['identity', 'binary_threshold', 'binary_random_draw'],
                p=[0.6, 0.3, 0.1]
            )
            measured_data[c] = self.measurement_func_options[measurement_func](measured_data[c])
        return measured_data

    def generate_r_output_value(self,
                                array=None,
                                node_number=None):
        # always use binary_threshold or binary_random_draw
        pass

    def get_options(self):
        return ('identity', 'scale')

    def get_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_options())
        )


class MissEdgeHandler:
    def __init__(self,
                 complexity=None,
                 zr_adj_matrix=None,
                 rr_adj_matrix=None,
                 xr_adj_matrix=None):
        # set the noise
        # limit options
        self.complexity = complexity

        self.zr_edge_func_options = {
            'identity': self.__identity,
            'linear': self.__linear,
            'sigmoid': self.__sigmoid,
            'sinusoidal': self.__sinusoid
        }
        self.xr_edge_func_options = {
            'identity': self.__identity,
            'linear': self.__linear,
            'sigmoid': self.__sigmoid,
            'sinusoidal': self.__sinusoid
        }
        self.rr_edge_func_options = {
            'attrition': self.__attrition,
            'exclusion': self.__exclusion,
            'random': self.__general
        }

        self.zr_edge_functions = None
        self.xr_edge_functions = None
        self.rr_edge_functions = None

        self._pick_zr_edge_functions(adj_matrix=zr_adj_matrix)
        self._pick_xr_edge_functions(adj_matrix=xr_adj_matrix)
        self._pick_rr_edge_functions(adj_matrix=rr_adj_matrix)

    def __attrition(self, array=None):
        return self.__general(array=array, attrition_coef=1)

    def __exclusion(self, array=None):
        return self.__general(array=array, attrition_coef=0)

    @staticmethod
    def __general(array=None, attrition_coef=None):
        if attrition_coef not in [0, 1]:
            attrition_coef = np.random.uniform()
        return [15 * (2 * attrition_coef * r - attrition_coef - r) for r in array]

    @staticmethod
    def __identity(array=None):
        return array

    @staticmethod
    def __linear(array=None):
        coef = np.random.normal(loc=1, scale=3)
        return array * coef

    @staticmethod
    def __sigmoid(array: np.ndarray = None):
        # 1/1+exp(k(c-x))
        # k in [-10, -1], [1, 10]
        # c a threshold in the data
        c = get_random_cut(array=array)
        k = np.random.choice([-1, 1]) * np.random.uniform(low=1, high=10)
        return [
            1/(1 + np.exp(k*(c-i))) for i in array
        ]

    @staticmethod
    def __sinusoid(array: np.ndarray=None):
        # a.sin(bx+c)
        # a in [-3, -1], [1, 3]
        # b in [1.6, 14]
        # c in [0, 2pi]
        scaled_array = minmax_scale(array)
        a = np.random.choice([-1, 1]) * np.random.uniform(low=1, high=3)
        b = np.random.uniform(low=1.5, high=14)
        c = np.random.uniform(low=0, high=2*np.pi)
        return a * np.sin(b * scaled_array + c)

    def _pick_zr_edge_functions(self, adj_matrix=None):
        print('adj matrix:', adj_matrix)
        num_nodes = adj_matrix.shape[0]
        print('num nodes:', num_nodes)
        self.edge_functions = np.random.choice(
            self.get_options(),
            size=(num_nodes, num_nodes),
            p=self.get_probs()
        )
        self.edge_functions = mask_matrix(
            matrix=self.edge_functions,
            mask=adj_matrix,
            mask_value='null'
        )
        return None

    def _pick_rr_edge_functions(self, adj_matrix=None):
        print('adj matrix:', adj_matrix)
        num_nodes = adj_matrix.shape[0]
        print('num nodes:', num_nodes)
        self.edge_functions = np.random.choice(
            self.get_options(),
            size=(num_nodes, num_nodes),
            p=self.get_probs()
        )
        self.edge_functions = mask_matrix(
            matrix=self.edge_functions,
            mask=adj_matrix,
            mask_value='null'
        )
        return None

    def _pick_xr_edge_functions(self, adj_matrix=None):
        print('adj matrix:', adj_matrix)
        num_nodes = adj_matrix.shape[0]
        print('num nodes:', num_nodes)
        self.edge_functions = np.random.choice(
            self.get_options(),
            size=(num_nodes, num_nodes),
            p=self.get_probs()
        )
        self.edge_functions = mask_matrix(
            matrix=self.edge_functions,
            mask=adj_matrix,
            mask_value='null'
        )
        return None

    def generate_z_input_values(self,
                              data=None,
                              parents=None,
                              node_number=None):
        inp_data = {p: None for p in parents}
        for p in parents:
            func = self.edge_functions[int(p[1:]), node_number]
            inp_data[p] = self.edge_func_options[func](array=data[p])

        return pd.DataFrame(inp_data)

    def generate_r_input_values(self,
                              data=None,
                              parents=None,
                              node_number=None):
        inp_data = {p: None for p in parents}
        for p in parents:
            func = self.edge_functions[int(p[1:]), node_number]
            inp_data[p] = self.edge_func_options[func](array=data[p])

        return pd.DataFrame(inp_data)

    def generate_x_input_values(self,
                              data=None,
                              parents=None,
                              node_number=None):
        inp_data = {p: None for p in parents}
        for p in parents:
            func = self.edge_functions[int(p[1:]), node_number]
            inp_data[p] = self.edge_func_options[func](array=data[p])

        return pd.DataFrame(inp_data)

    def get_zr_options(self):
        return (
            'identity', 'linear',
            'sigmoid', 'sinusoidal'
        )

    def get_rr_options(self):
        return (
            'identity', 'linear',
            'sigmoid', 'sinusoidal'
        )

    def get_xr_options(self):
        return (
            'identity', 'linear',
            'sigmoid', 'sinusoidal'
        )

    def get_z_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_options())
        )

    def get_r_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_options())
        )

    def get_x_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_options())
        )


class MissStateHandler:
    pass


class MissOutputHandler:
    pass