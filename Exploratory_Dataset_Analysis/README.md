# Exploratory Dataset Analysis (EDA)

An analysis was conducted on the selected subset to examine the distribution of music styles and the classification of audio files according to these styles. This exploratory analysis allowed for:

- Assessment of the balance of the subset across different music styles.
- Identification of potential biases in the dataset that could affect model performance.
- Insights into style prevalence, informing decisions for preprocessing and model evaluation.


## Installation
Create an environment with the necessary dependencies. Python 3.10 is recommended.

Using pip: 
```
pip install -r requirements.txt
```

Or, if using conda:
```
conda env create -f flg_env.yml
conda activate flg_env
python -m ipykernel install --user --name=flg_env --display-name "Python (flg_env)"

```

## Usage
1. Calculate style embeddings for each file, along with readable top labels, dictionaries, and probabilites:
    ````
    python calculate_style_distribution.py
    ````
2. Classify the files by top genre:
    ````
    python genre_classification.py
    ````

3. Run the cells in `Exploratory_Dataset_Analysis.ipynb` to get the dataset genre distribution analysis.

