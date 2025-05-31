# Root Join (Task 2)

This repository is based on the sisap25-example-python repository and contains a Python implementation of the Root Join algorithm (as described in this paper), combined with various techniques designed to solve Task 2 of the SISAP 2025 Indexing Challenge.


## Step for running

A working installation of Python 3.12 is required, along with the following libraries: Numpy, H5py, Scikit-learn, and Pandas.

The steps are the following:

1. Clone the repository
2. Install requirements
3. Run
4. Evaluate

The full set of installation instructions are listed in the GitHub Actions workflow

## Clone the repository

bash
git clone https://github.com/Fa11ingDeep/SISAP-Challenge-2025.git
cd SISAP-Challenge-2025

## Install the requirements

bash
pip install -r requirements.txt 

## Run

Run the second task on an example input (by default is set with gooaq dataset)
It will automatically take care of downloading the necessary example dataset.

bash
python root_join.py

or

bash
python root_join.py --task task2 --dataset gooaq

## Evaluation

will produce a summary file of the results with the computed recall against the ground truth data.

```bash
python eval.py results.csv
