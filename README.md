# People annual incoming predictor

![Conda](https://img.shields.io/conda/pn/conda-forge/python)
![licence](https://img.shields.io/badge/language-Python-brightgreen.svg?style=flat-square)

This project implements an API that can infer if one person can gain more than 50k at year.

On this project the CI/CD paradigm is implemented

- GitHub release 1.0.0: <https://github.com/BolivarTech/census_data_processor.git>

The following tools are used:

- [DVC](https://dvc.org) for artifacts tracking and management.
- [GitHub](https://github.com/BolivarTech/census_data_processor) For Continuous Integration (CI).
- [Render](https://www.render.com/) for Continuous Deployment (CD).
- [Conda](https://docs.conda.io/en/latest/) for environment management.
- [Pandas](https://pandas.pydata.org) for data analysis.
- [Scikit-Learn](https://scikit-learn.org/stable/) for data modeling.
- [Postman](https://https://www.postman.com/) for API testing.

The final deployment can found at [Census Data Processor](https://census-data-processor.onrender.com)

## How to Use This Project

1. Install the [dependencies](#dependencies).
2. Access the [API](#how-to-access-the-api) section.

### Dependencies

In order to set up the main environment from which everything is launched you need to install [conda](https://docs.conda.io/en/latest/) and the following sets everything up:

```bash
# Clone repository
git clone https://github.com/BolivarTech/census_data_processor.git
cd census_data_processor

# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate census_processor
```

If you need to execute the EDA notebook, follow the next steps.

```bash
# move to eda directory
cd census_data_processor/eda

# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate census-eda

# Run Jupyter-Lab
jupyter-lab
```

### How to access the API

The easy way to test the API is using the docs and test the samples at:

<https://census-data-processor.onrender.com/docs>

On the documentation the API usage can be found.

**Note:** because this is a test environment the and is not fully traffic on it, the
service maybe stopped due inactivity and when it is accessed by first time, it look
some time to restart the service and it can look irresponsive for a while.

## Authorship

[Julian Bolivar](https://www.linkedin.com/in/jbolivarg), 2023.  
