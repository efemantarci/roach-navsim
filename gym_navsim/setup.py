from setuptools import setup, find_packages
setup(
    name="gym_navsim",
    version="0.0.1",
    install_requires=["gym==0.17.2"],
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
)