# nlp-rule-extraction
Deliverable for topic B of the TUM WS2023 Master course "Approaching Information System Challenges with Natural Language Processing".

For questions please contact leo.holzhauer@tum.de.

## Quick Setup

1. Clone repo.
2. Setup venv conda.
3. Run requirements.txt.
4. Download spacy packages with this command in terminal: `python -m spacy download en_core_web_lg`

During the execution of the script, a pretrained spacy pipeline is used for various tasks, such as tokenization. It is automatically downloaded, if not yet present in your local environment. The  default is `en_core_web_lg` (large), which takes up to 600 MB of storage. Alternatives for reduced memory usage are `en_core_web_md` (medium) and `en_core_web_sm` (small). However, these might reduce model performance. 

## Recommendation for Input Data

1. Demark sections with '#' and subsections with '##'. 

## Overview

### Used Data Sources

To realize a bigger sample size for training, different data sources are combined. The files with the prefix *input* are the textual descriptions from which the constraints shall be extracted. The files with the prefix *output* are the expected constraints as a result of the NLP model, also refered to as Golden Standard (GS).

#### Coffee roasting
Synthethic data describing a coffee roasting process. It contains the following files:
1. *input-coffee.txt*: textual description of the coffee roasting process, serving as an input to the model.
2. *output-coffee.txt*: extracted constraints from *input-coffee.txt* in a structured form. 
3. *coffee_process.png*: visualisation of the coffee roasting process.
4. *coffee_constraints.png*: visualisation of the constraints.

#### CDM

#### AktG

#### PatG

## Known bugs






