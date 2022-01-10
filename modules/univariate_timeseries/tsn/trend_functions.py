def function_lookup(function_name):
    table = {
        'linear': linear_function
    }
    return table[function_name]


def function_params_lookup(function_name):
    table = {
        'linear': [
            {'name': 'trend_slope', 'range': [-2, 2]},
            {'name': 'trend_intercept', 'range': [0, 10]}
        ]
    }
    return table[function_name]


def linear_function(trend_slope=None, trend_intercept=None, t=None):
    return trend_slope * t + trend_intercept
