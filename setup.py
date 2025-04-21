from setuptools import setup, find_packages

setup(
    name="my_ml_package",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'scikit-learn',
        'joblib',
        'numpy'
    ],
)