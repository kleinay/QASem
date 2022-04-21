# QASem - Question-Answer based Semantics 



## Pre-requisite
* Python 3.7

## Installation
From pypi:
```bash
pip install transformers==4.15.0 spacy==2.3.7 qanom 
pip install git+https://github.com/rubenwol/RoleQGeneration.git
python -m spacy download en_core_web_sm
```

If you want to install from source, clone this repository and then install requirements:
```bash
git clone https://github.com/kleinay/QANom.git
cd QANom
pip install requirements.txt
cd ..
git clone https://github.com/rubenwol/RoleQGeneration.git
```

## End-to-End Pipeline 

If you wish to parse sentences with QASRL and QANom, the best place to start is the `QASemEndToEndPipeline` from the python file [end_to_end_pipeline.py](https://github.com/kleinay/QASem/blob/main/qasem/end_to_end_pipeline.py). 

Note: Soon, we will also combine QADiscourse (Pytakin et. al., 2020) and other ongoing-work layers of QA-based semantic annotations for adjectives and noun modifiers. 


**Usage Example**

 ```python
from qasem.end_to_end_pipeline import QASemEndToEndPipeline 
pipe = QASemEndToEndPipeline(detection_threshold=0.75)  
sentences = ["The doctor was interested in Luke 's treatment .", "Tom brings the dog to the park."]
outputs = pipe(sentences, return_detection_probability = True,
                 qasrl = True,
                 contextual_qasrl = True,
                 qanom = True,
                 contextual_qanom = True)

print(outputs)
 ```
Outputs
 ```python
[{'qanom': [
   {'QAs': [{
      'question': 'who was treated ?',
      'answers': ['Luke'],
      'contextual_question': 'Who was treated?'}],
    'predicate_idx': 7,
    'predicate': 'treatment',
    'predicate_detector_probability': 0.8152085542678833,
    'verb_form': 'treat'}],
  'qasrl': []},
  
 {'qanom': [],
  'qasrl': [{'QAs': [
     {'question': 'who brings something ?',
      'answers': ['Tom'],
      'contextual_question': 'Who brings the dog?'},
     {'question': ' what does someone bring ?',
      'answers': ['the dog'],
      'contextual_question': 'What does Tom bring?'},
     {'question': ' where does someone bring something ?',
      'answers': ['to the park'],
      'contextual_question': 'Where does Tom bring the dog?'}],
    'predicate_idx': 1,
    'predicate': 'brings',
    'verb_form': 'bring'}]}]
 ```
