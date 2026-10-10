from setuptools import find_packages, setup

setup(
    name="liveanon",
    version="0.1.0",
    description="LiveAnon: Streamable Speaker Anonymization",
    license="Apache-2.0",
    packages=find_packages(include=["liveanon", "liveanon.*"]),
    python_requires=">=3.10",
)
