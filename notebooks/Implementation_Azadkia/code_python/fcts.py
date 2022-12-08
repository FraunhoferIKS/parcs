import numpy as np

def arctanx2(data):
    return np.arctan(data['X2'])

def squarex3(data):
    return data['X3']**2

def sinx3(data):
    return np.sin(data['X3'])

def computex9(data):
    return np.sin(data['X6'] + data['e9']) + np.abs(data['X10'])

def sinx11(data):
    return np.sin(data['X11'])

def sinx12(data):
    return np.sin(data['X12'])

def abssqrtx12(data):
    return np.sqrt(np.abs(data['X12']))

def computex13(data):
    return np.arctan(data['X9']**2 + data['e13'])
