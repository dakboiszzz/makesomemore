from setuptools import setup, find_packages

setup(
    name="makesomemore",
    version="0.1.0",
    packages=find_packages(), # This automatically finds 'src' and sub-packages
    install_requires=[
        "torch",
        "numpy",
        "fastapi",
        "uvicorn",
        "pyyaml"
    ],
)