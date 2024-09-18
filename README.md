# Rotordynamics UFU
Repo to store rotordynamic codes and computation results developed in the context of my master studies

## Note to package it

Run the following commands to package it:

```
python setup.py sdist bdist_wheel
```

Them, you can install it with:

```
pip install -e ".[dev]"
```

## Note to environment

The versions of the packages in this repo is saved in `requirements.txt`.

However, it is important to note that in this project we use Conda to manage the environmentals. Therefore to create the environment from the YAML file: Open your terminal or command prompt and run the following command to create the environment:

```
conda env create -f environment.yml
```