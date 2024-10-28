# GATMobilityAnalysis
Applying GAT and LSTM to understand human mobility patterns and anomalies

## Environment Setup

We use Conda to manage dependencies. The `environment.yml` file located in the repository specifies all the required packages for this project.

### Create the Conda Environment from environment.yml

To create the environment using the provided environment.yml file, run the following command:

```
conda env create -f environment.yml
```

This will automatically create a Conda environment named deep-learning and install all required dependencies, including Python 3.9.19.

### Activate the Environment

After creating the environment, activate it with the following command:

```
conda activate deep-learning
```

### Add the Environment as a Jupyter Kernel

To use this environment in Jupyter notebooks, you need to add it as a Jupyter kernel:

```
python -m ipykernel install --user --name=deep-learning --display-name "Python (deep-learning)"
```

Once done, you can select the Python (deep-learning) kernel in Jupyter notebooks.
