from setuptools import setup, find_packages

setup(
    name="ml_package",  # <- Новое имя
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ml_package": ["model/*.joblib"],  # <- Обновленный путь
    },
    install_requires=[
        'scikit-learn',
        'joblib',
        'numpy'
    ],
)