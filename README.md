# QASem - Question-Answer based Semantics 

This repository includes software for parsing natural language sentence with various layers of QA-based semantic annotations. 
We currently support three layers of semantic annotations - QASRL, QANom, and QADiscourse. 
See an overview of our approach at our paper on [QASem Parsing](https://arxiv.org/abs/2205.11413). 

[QASRL (Question Answer driven Semantic Role Labeling)](https://aclanthology.org/D15-1076/) is a lightweight semantic framework for annotating "who did what to whom, how, when and where". 
For every verb in the sentence, it provides a set of question-answer pairs, where the answer mark a participant of the event denoted by the verb, while the question captures its *semantic role* (that is, what is the role of the participant in the event).

"QANom" stands for "QASRL for Nominalizations", which is an adaptation of QASRL to (deverbal) nominalization. See the [QANom paper](https://aclanthology.org/2020.coling-main.274/) for details about the task. 

You can find more information on [QASRL's official website](https://qasrl.org), including links to all the papers and datasets and a data browsing utility. 
We also wrapped the datasets into Huggingface Datasets ([QASRL](https://huggingface.co/datasets/kleinay/qa_srl); [QANom](https://huggingface.co/datasets/biu-nlp/qanom)), which are easier to plug-and-play with (check out our [HF profile](https://huggingface.co/biu-nlp) for other related datasets, such as QAMR, QADiscourse, and QA-Align).

[QADiscourse](https://aclanthology.org/2020.emnlp-main.224) annotates intra-sentential discourse relations with question-answer pairs. It focus on discourse relations that carry information, rather than specifying structural or pragmatic properties of the realied sentencs. Each question starts with one of 17 crafted question prefixes, roughly mapped into PDTB relation senses.   

*Note*: In the future, we will also combine additional layers of QA-based semantic annotations for adjectives and noun modifiers, currently at the stage of ongoing work. 


## Demo

Check out the [live QASem demo](https://huggingface.co/spaces/kleinay/qasem-demo) on Huggingface.




## Installation

**Pre-requisite**: Python 3.7

Installation is available via pip:
```bash
pip install qasem
```

### Installation from source
Clone the repo and install using `setup.py`:
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

The `QASemEndToEndPipeline` class would, by demand, parse sentences with any of the QASem semantic annotation layers --- currenlty including 'qasrl', 'qanom' and 'qadiscourse'.  

### Features

**Run on GPU:**
Use `device=d` in initialization to put models and tensors on a GPU device, where `d` is the CUDA device ID. We currently do not support parallelization on multiple GPUs. Defaults to `device=-1`, i.e. CPU.  

**Annotation layers:**
By default, the pipeline would parse all layers.
To specify a subset of desired layers, e.g. QASRL and QADiscourse alone, use `annotation_layers=('qasrl', 'qadiscourse')` in initialization.

**QA-SRL contextualization:**
For the sake of generality, QA-SRL and QANom generate ``abstractive'' questions, that replace arguments with placeholders, e.g. "Why was *someone* interested in *something*?". However, in some use-cases you might want to have a more natural question with contextualized arguments, e.g. "Why was *the doctor* interested in *Luke 's treatment*?". Utilizing the model from [Pyatkin et. al., 2021](https://aclanthology.org/2021.emnlp-main.108/), one can additionally get contextualized questions for QA-SRL and QANom by setting `QASemEndToEndPipeline(contextualize=True)` (see example below).     

**QA-SRL Discrete Roles:** In QA-SRL, semantic roles are captured in a rich but soft manner within the questions. For some applications, a reduced discrete account of semantic roles may be desired. By default (`return_qasrl_discrete_role=True` in initialization), we provide a discrete "question-role" label per question in the output, based on a heuristic mapping from the question syntactical structure. For the core arguments, "R0" corresponds to asking about the subject position (commonly equivalent to proto-agent semantic roles), "R1" to direct object (proto-patient), "R2" to a second direct object, and "R2_<preposition>" to an indirect object (e.g. "R2_on" <-> "what did someone put something *on*?"). For modifiers ("where", "when", "how", "why", "how long", "how much") the WH-word (plus, optionally, the preposition) is defining the "question-role". See Table 7 at the [QA-SRL 2015 paper](https://dada.cs.washington.edu/qasrl/docs/emnlp2015_hlz.pdf) for more details about the set of Roles and the heuristic mapping.  

**QA-SRL Question slots:** Set `return_qasrl_slots=True` in initialization to get detailed information about each QA-SRL question. This includes the 7 slots comprising the question, the verb inflection, voice ("is_passive") and negation ("is_negated"). 

**Nominal predicate detection:**
`nominalization_detection_threshold` --- which can be set globally in initialization and per `__call__` --- is the threshold for the nominalization detection model.
A higher threshold (e.g. `0.8`) means capturing less nominal predicates with higher confidence of them being, in context, verb-derived event markers. Default threshold is `0.7`. 

**OpenIE converter:**
Set `output_openie=True` (in `__call__`) in order to get a reduction of output QAs into Open Information Extraction's tuples format. This option uses the `qasem.openie_converter.OpenIEConverter` class to linearize the arguments along with the predicate by the order of occurrence in the source sentence. 
The pipeline's output would then be in the form `{"qasem": <regular QA outputs>, "openie": <OpenIE tuple outputs>}`.

By default, only verbal QA-SRL QAs would be converted, but one can specify `layers_included=["qasrl", "qanom"]` when initializing `OpenIEConverter` to also include nominalizations' QAs. 
You can set arguments for `OpenIEConverter` in the `QASemEndToEndPipeline` constructor using the `openie_converter_kwargs` argument, e.g. `QASemEndToEndPipeline(openie_converter_kwargs={"layers_included": ["qasrl", "qanom"]})`. 


### Example

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


## Repository for Model Training & Experiments

The underlying QA-SRL and QANom models were trained and evaluated using the code at [qasrl-seq2seq](https://github.com/kleinay/qasrl-seq2seq) repository.

The code for training and evaluating the QADiscourse model will be uploaded soon.

## Cite

```latex
@article{klein2022qasem,
  title={QASem Parsing: Text-to-text Modeling of QA-based Semantics},
  author={Klein, Ayal and Hirsch, Eran and Eliav, Ron and Pyatkin, Valentina and Caciularu, Avi and Dagan, Ido},
  journal={arXiv preprint arXiv:2205.11413},
  year={2022}
}
```
