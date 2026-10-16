from setuptools import setup, find_namespace_packages

NAME = "mesh-opformer-geodesic"
VERSION = "0.0.1"
DESCRIPTION = (
    "Geodesic-ring sparse attention for HodgeFormer: multi-scale, hop-stratified "
    "neighbourhoods + relative-geodesic positional bias."
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author="",
    author_email="",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        # numeric / graph substrate is provided by the sibling packages, which
        # are expected to be installed editable in the same env:
        #   mesh_sim        -> BFS levels (graphblas)
        #   mesh_opformer   -> gather kernels reused by the ring attention
        "numpy",
        "torch",
    ],
    python_requires=">=3.10",
)
