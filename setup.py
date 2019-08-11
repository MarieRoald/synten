from setuptools import setup, find_packages


setup(
    name="synthetic_tensors",
    author="Marie Roald"
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
)
