from setuptools import setup

setup(
    name="biofts",
    description="A python package for field theoretic simulations of biopolymer solutions.",
    author="Jonas Wessén",
    version="0.1",
    packages=["biofts"],
    install_requires=[
        # List any dependencies your package requires
        "numpy"
    ],
)
