# nlp-rule-extraction
Deliverable for topic B of the TUM WS2023 Master course "Approaching Information System Challenges with Natural Language Processing".

This provides a rule-based structure to 

For questions please contact leo.holzhauer@tum.de.

## Quick Setup

1. Clone repo.
1. Setup venv conda.
1. Run requirements.txt. 

During the execution of the script, a pretrained spacy pipeline is used for various tasks, such as tokenization. It is automatically downloaded, if not yet present in your local environment. 

The  default is `en_core_web_lg` (large), which takes up to 600 MB of storage. Alternatives for reduced memory usage are `en_core_web_md` (medium) and `en_core_web_sm` (small). However, these were not tested and might reduce performance. 

For more information, please refer to the [spaCy documentation](https://spacy.io/models/en).

## Repository Structure

```plaintext
nlp-rule-extraction/
├── data/
├── notebooks/
│   └── pipeline.ipynb
    └── plots/
├── reports/
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── modelling.py
│   ├── preprocess.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Usage

### Recommendations on Input Data

1. Demark sections with '#' and subsections with '##'. 
1. For enumeration items, restrict yourself to items of the following class 1., (1), (a) and (i). Each enumeration should start with a describtion and a colon (" : ") and there should be a linebreak preceeding each enumeration item.

## Data Sources

To realize a bigger sample size, different data sources are combined. The files with the prefix *input* are the textual descriptions from which the constraints shall be extracted. The files with the prefix *output* are the expected constraints as a result of the NLP model, also refered to as Golden Standard (GS).

### Coffee roasting
Synthethic data describing a coffee roasting process. It contains the following files:
1. *input-coffee.txt*: textual description of the coffee roasting process, serving as an input to the model.
1. *output-coffee.txt*: extracted constraints from *input-coffee.txt* in a structured form. 
1. *coffee_process.png*: visualisation of the coffee roasting process.
1. *coffee_constraints.png*: visualisation of the constraints.

### CDM

### AktG

### PatG

## Acknowledgements

I want to thank Catherine Sai for her supervision and mentoring and Jasper Schmieding for his help in collecting the data .





