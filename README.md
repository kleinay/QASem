# QASem - Question-Answer based Semantics 

This repository includes software for parsing natural language sentence with various layers of QA-based semantic annotations. 
We currently support three layers of semantic annotations - QASRL, QANom, and QADiscourse.

[QASRL (Question Answer driven Semantic Role Labeling)](https://aclanthology.org/D15-1076/) is a lightweight semantic framework for annotating "who did what to whom, how, when and where". 
For every verb in the sentence, it provides a set of question-answer pairs, where the answer mark a participant of the event denoted by the verb, while the question captures its *semantic role* (that is, what is the role of the participant in the event).

"QANom" stands for "QASRL for Nominalizations", which is an adaptation of QASRL to (deverbal) nominalization. See the [QANom paper](https://aclanthology.org/2020.coling-main.274/) for details about the task. 

You can find more information on [QASRL's official website](https://qasrl.org), including links to all the papers and datasets and a data browsing utility. 
We also wrapped the datasets into Huggingface Datasets ([QASRL](https://huggingface.co/datasets/kleinay/qa_srl); [QANom](https://huggingface.co/datasets/biu-nlp/qanom)), which are easier to plug-and-play with (check out our [HF profile](https://huggingface.co/biu-nlp) for other related datasets, such as QAMR, QADiscourse, and QA-Align).

[QADiscourse](https://aclanthology.org/2020.emnlp-main.224) annotates intra-sentential discourse relations with question-answer pairs. It focus on discourse relations that carry information, rather than specifying structural or pragmatic properties of the realied sentencs. Each question starts with one of 17 crafted question prefixes, roughly mapped into PDTB relation senses.   

*Note*: Soon, we will also combine additional layers of QA-based semantic annotations for adjectives and noun modifiers, currently at the stage of ongoing work. 



## Pre-requisite
* Python 3.7

## Installation

We will soon release a first version to pypi.
But meantime, the simplest way to get it work is to clone the repo and install using `setup.py`:
```bash
git clone https://github.com/kleinay/QASem.git
cd QASem
pip install -e .
```

Alternatively, If you want to install the dependencies explicitly:
```bash
pip install transformers==4.15.0 spacy>=2.3.7 qanom 
pip install git+https://github.com/rubenwol/RoleQGeneration.git
```

In addition, you would need to download a spacy model for pre-requisite tokenization & POS-tagging:
```bash
python -m spacy download en_core_web_sm
```



## Usage 

The `QASemEndToEndPipeline` class would, by demand, parse sentences with any of the QASem semantic annotation layers --- currenlty including 'qasrl', 'qanom' and 'qadiscourse'. By default, the pipeline would parse all layers. 
To specify a subset of desired layers, e.g. QASRL and QADiscourse alone, use `annotation_layers=('qasrl', 'qadiscourse')` in initialization.

**Example**

 ```python
from qasem.end_to_end_pipeline import QASemEndToEndPipeline 
pipe = QASemEndToEndPipeline(annotation_layers=('qasrl', 'qanom', 'qadiscourse'),  nominalization_detection_threshold=0.75, contextualize = True)  
sentences = ["The doctor was interested in Luke 's treatment as he was still not feeling well .", "Tom brings the dog to the park."]
outputs = pipe(sentences)

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
    'verb_form': 'treat'}
  ],
  'qasrl': [
    ...
  ],
  'qadiscourse': [{
    'question': 'What is the cause of the doctor being interested in Luke 's treatment?',
    'answer': 'he was still not feeling well'}
  ]},
 },
 
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
    'verb_form': 'bring'}]}
  ],
  'qadiscourse': []
 }
 ```

`nominalization_detection_threshold` is the threshold for the nominalization detection model, where a higher threshold (e.g. `0.8`) means capturing less nominal predicates with higher confidence of them being, in context, verb-derived event markers. Default threshold is `0.7`. 



## Demo

Check out the [live demo for our joint QASRL-QANom model](https://huggingface.co/spaces/kleinay/qanom-seq2seq-demo)!

If you wish to test the nominalization detection component, see [its own demo here](https://huggingface.co/spaces/kleinay/nominalization-detection-demo), 
or visit the [QANom End-To-End demo](https://huggingface.co/spaces/kleinay/qanom-end-to-end-demo).



## Repository for Model Training & Experiments

The underlying QA-SRL and QANom models were trained and evaluated using the code at [qasrl-seq2seq](https://github.com/kleinay/qasrl-seq2seq) repository.

The code for training and evaluating the QADiscourse model will be uploaded soon.
