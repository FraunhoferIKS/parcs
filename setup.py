from setuptools import setup, find_packages

setup(
    name='rad_sim',
    version='0.0.1',
    description='Simulation package',
    author='RAD-IKS@FhG',
    author_email='alireza.zamanian@iks.fraunhofer.de',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['numpy', 'dtaidistance~=2.3.6'],
    classifiers=[
        'Development Status :: First Internal Release',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
