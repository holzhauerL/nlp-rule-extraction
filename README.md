# nlp-rule-extraction
Deliverable for topic B of the TUM WS2023 Master course "Approaching Information System Challenges with Natural Language Processing".

For questions please contact leo.holzhauer@tum.de.

## Quick Setup

1. Clone repo.
2. Setup venv conda.
3. Run requirements.txt.
4. Download spacy packages:
    - python -m spacy download en_core_web_sm
    - python -m spacy download en_core_web_md
    - python -m spacy download en_core_web_lg

## Overview

### Used Data Sources

To realize a bigger sample size for training, different data sources are combined. The files with the prefix *input* are the textual descriptions from which the constraints shall be extracted. The files with the prefix *output* are the expected constraints as a result of the NLP model, also refered to as Golden Standard (GS).

#### Coffee roasting
Data provided by the supervisor of the course describing a coffee roasting process. It contains the following files:
1. *input-coffee.txt*: textual description of the coffee roasting process, serving as an input to the model.
2. *output-coffee.txt*: extracted constraints from *input-coffee.txt* in a structured form. 
3. *coffee_process.png*: visualisation of the coffee roasting process.
4. *coffee_constraints.png*: visualisation of the constraints.

## Known bugs






