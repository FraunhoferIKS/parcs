import numpy
import pytest
import os
from pyparcs.api.graph_objects import *


class TestConstantNodes:
    @staticmethod
    @pytest.mark.parametrize('value', [1, np.int64(-1), np.float32(2.5), -2.5, 0.4, -0.4, 0, -0])
    def test_constant_node(value):
        node = ConstNode(value=value)
        result = node.calculate(size_=4)
        assert np.array_equal(result, [value, value, value, value])

    @staticmethod
    @pytest.mark.parametrize('value', ['A', None])
    def test_constant_node_raise_errors(value):
        with pytest.raises(TypeError):
            ConstNode(value=value)


class TestDataNodes:
    @staticmethod
    @pytest.fixture(scope='class')
    def csv_file():
        # setup
        df = pd.DataFrame({
            'Z_1': [1, 2, 3, 4]
        })
        file_name = 'test_file.csv'
        df.to_csv(file_name)
        # test
        yield file_name
        # teardown
        os.remove(file_name)

    @staticmethod
    @pytest.mark.parametrize('err,realization', [
        ([0.01, 0.1, 0.2], [1, 1, 1]),
        ([0.26, 0.4, 0.48], [2, 2, 2]),
        ([0.51, 0.70, 0.74], [3, 3, 3]),
        ([0.76, 0.85, 0.99], [4, 4, 4]),
    ])
    def test_data_node(err, realization, csv_file):
        node = DataNode(csv_dir=csv_file, col='Z_1')
        # error term [0.0, 0.3, 0.6, 0.9] gives 1, 2, 3, 4
        out = node.calculate(pd.Series(err))
        assert numpy.array_equal(out, realization)

    @staticmethod
    @pytest.mark.parametrize('array,error', [
        (np.array([0, 0.2]), TypeError),  # not pandas series
        (pd.Series(['a', 'b']), ValueError),  # non-numeric values
        (pd.Series([0, 0.4, 1.1]), ValueError),  # value greater than 1
        (pd.Series([-0.001, 0.2]), ValueError)  # value smaller than zero
    ])
    def test_data_node_raises_error(array, error, csv_file):
        with pytest.raises(error):
            node = DataNode(csv_dir=csv_file, col='Z_1')
            node.calculate(array)


class TestDeterministicNode:
    @staticmethod
    def test_deterministic_node():
        node = DetNode(function=lambda d: d['Z_1'] + d['Z_2'])
        data = pd.DataFrame({
            'Z_1': [1, 2, 3],
            'Z_2': [2, 3, 3],
            'Z_3': [0, 1, 2]
        })
        out = node.calculate(data)
        assert numpy.array_equal(out, [3, 5, 6])

    @staticmethod
    @pytest.mark.parametrize('func,data,error', [
        (lambda d: d['Z_1'], np.array([[1, 2], [2, 1]]), TypeError),  # data is not pandas data frame
        (lambda: 1, pd.DataFrame({'Z_1': [1, 2]}), ExternalResourceError),  # function doesn't have inputs
        (lambda a, b: 1, pd.DataFrame({'Z_1': [1, 2]}), ExternalResourceError),  # function has more than 1 inputs
        (lambda d: 3*d['Z_1'], pd.DataFrame({'Z_2': [1, 2]}), ExternalResourceError),  # parents names do not match
    ])
    def test_data_node(func, data, error):
        with pytest.raises(error):
            node = DetNode(function=func)
            node.calculate(data)
