from setuptools import setup,find_packages
setup(
    name="RBM",
    version="1.0.7",
    author="Shinsuke Sakai",
    url="https://github/ShinsukeSakai0321/RBM.git",
    packages=find_packages("src"),
    package_dir={"":"src"},
    package_data={'':['*.csv']},
    include_package_data=True,
)