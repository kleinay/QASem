
# Role Questions
This repository contains the official implementation of the method described in: [Asking It All: Generating Contextualized Questions for Any Semantic Role](http://https://aclanthology.org/2021.emnlp.XXXX)
by [Valentina Pyatkin](https://valentinapy.github.io) (BIU), [Paul Roit](https://paulroit.com) (BIU), [Julian Michael](https://julianmichael.org), [Reut Tsarfaty](https://nlp.biu.ac.il/~rtsarfaty/) (BIU, AI2) [Yoav Goldberg](https://u.cs.biu.ac.il/~yogo/) (BIU, AI2) and [Ido Dagan](https://u.cs.biu.ac.il/~dagani/) (BIU) 

Paper Abstract:
> Asking questions about a situation is an inherent step towards understanding it. 
> To this end, we introduce the task of role question generation, which, given a predicate mention and a passage, requires producing a set of questions asking about all possible semantic roles of the predicate. We develop a two-stage model for this task, which first produces a context-independent question prototype for each role and then revises it to be contextually appropriate for the passage. 
> Unlike most existing approaches to question generation, our approach does not require conditioning on existing answers in the text. 
> Instead, we condition on the type of information to inquire about, regardless of whether the answer appears explicitly in the text, could be inferred from it, or should be sought elsewhere. 
> Our evaluation demonstrates that we generate diverse and well-formed questions for a large, broad-coverage ontology of predicates and roles.

## Introduction
This paper presents a method to generate questions that inquire about certain semantic concepts that may appear in a text.  
For example, in the text: 
> John **sold** the pen to Mary 

the predicate word sold evokes a semantic frame with the
following roles fulfilled by the verb's explicit arguments: _John_ as <u>the seller</u>, _the pen_ as <u>the thing sold</u> and _Mary_ as <u><the buyer</u>.
The predicate sell also evokes another semantic role which is not fulfilled in the context of this sentence: <u>The price paid</u>.
In this work we would like to create grammatical, fluent and fit-to-context questions that target each such role, whether it is fulfilled or not.

Given the source sentence and the target role, we would like to create the following questions:

* John **sold** the pen to Mary; <u>The seller</u> ==> Who sold the pen to Mary?
* John **sold** the pen to Mary; <u>The buyer</u> ==> Who did John sell the pen to?
* John **sold** the pen to Mary; <u>The thing sold</u> ==> What did John sell?
* John **sold** the pen to Mary; <u>The price paid</u> ==> What did John sell the pen for?

This work relies on a semantic ontology to list and identify all semantic roles, and is implemented on top of [PropBank](https://github.com/propbank/).

If you simply want to predict questions using our method, check the installation requirements and the "Easy Way to Predict Role Questions" section.
For reproducibility reasons we also detail how to obtain various steps of our pipeline (you don't need to follow these if you only want to use the model for inference).

## Installation Requirements:
The following python libraries are required:
 - torch==1.7.1
 - spacy==2.3.2
 - transformers==4.1.1
 - allennlp==1.2.0rc1
This project uses data and code from: the [QA based Nominal SRL project](https://github.com/kleinay/QANom) 
This project (QANom) can be installed with pip (pip install qanom).

## Easy Way to Predict Role Questions:
If you just want to predict Role Questions for a given context and predicate(s), you can use a simple script that we prepared.
You should download and unzip the [contextualizer model](https://nlp.biu.ac.il/~pyatkiv/roleqsmodels/question_transformation.tar.gz) .
To run the script you can use the following command: 
> python predict_questions.py --infile <INPUT_FILE_PATH> --outfile <OUTPUT_FILE_PATH> --transformation_model_path <PATH_TO_DOWNLOADED_CONTEXTUALIZER_MODEL> --device_number <NUMBER_OF_CUDA_DEVICE> --with_adjuncts <TRUE or FALSE>

The input file should be a jsonl file (check debug_file.jsonl for an example), containing the following information: an instance id (id),
the sentence the target predicate appears in (sentence), the target index of the predicate in the sentence (target_idx), the target lemma (target_lemma),
the POS of the target (target_pos), the predicate sense in terms of OntoNotes (predicate_sense). Our model works best with disambiguated predicate senses
in terms of OntoNotes, so if you have a predicate disambiguation system or gold sense information please include it. Otherwise you could simply choose the first
sense by putting 1 in that field, with some performance tradeoffs.


## Data Dependencies
The following datasets are required in-order to re-create our data and evaluation. 
Our scripts will refer to the root directories after downloading or after extracting these resources. 
- For collecting prototypes and training the contextualizer:
  -- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) (The frame files under English metadata) _Download from the LDC_
  -- [OntoNotes 5.0 (2012) in CoNLL format](https://github.com/ontonotes/conll-formatted-ontonotes-5.0/). _Please convert the gold skeleton files to gold conll files_
  -- [QA-SRL Bank 2.0](http://qasrl.org/data/qasrl-v2.tar). 
  -- [QANom](https://github.com/kleinay/QANom)
  -- [NomBank 1.0](https://nlp.cs.nyu.edu/meyers/nombank/nombank.1.0.tgz)
  -- [PennTreeBank v3](https://catalog.ldc.upenn.edu/LDC99T42) _Download from the LDC_
- For evaluation:
  -- [Gerber and Chai](http://lair.cse.msu.edu/projects/implicit_annotations.html) 
  -- [ON5V](http://projects.cl.uni-heidelberg.de/india/)
  
## Model Dependencies
We have used the publicly available verbal SRL parser by AllenNLP, the download link is:
[bert-base-srl-2020.03.24](https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz)
AllenNLP may change storage or re-train this model with version changes, the latest can be found here: [SRL in AllenNLP](https://demo.allennlp.org/semantic-role-labeling)
We are also extensively using BERT and BART models via the huggingface python library, but they aren't explicitly mentioned due to their ubiquitousness.
 
 
## Data Preprocessing Instructions 
All scripts are run from the root directory of this project. 
#### OntoNotes
We preprocess OntoNotes into a json-lines format, where each output line contains a sentence,
its frames (predicate and arguments). Given co-ref data we enrich each frame with other mentions  
of explicit arguments from any sentence in the document. 
These would be marked as implicit arguments bearing the same role as their explicit mentions. 
Note that more implicit arguments could be present in the text, but they are unmarked in OntoNotes.
  
> python ./ontonotes/preprocess_ontonotes.py --conll_onto_path <CONLL_FORMATTED_ONTONOTES_DIR>

This will generate ./ontonotes/ontonotes.[train|dev|test].jsonl

#### QA-SRL and QANom

> python ./qasrl/preprocess_qasrl_bank.py --qasrl_path <QASRL_V2_DIR>

> pyhton ./qasrl/preprocess_qanom --qanom_dir <QA_NOM_DIR>


#### Frames

We process the PropBank frame files that were distributed with the OntoNotes 5.0 release in 2012.
 Since then PropBank frame files had another large release (The Unified PropBank of 2016) 
 but to stay compatible with the predicate-sense annotation in OntoNotes we refrain from using it. 
 
> python ./role_lexicon/preprocess_ontonotes_roles.py --onto_dir <ONTONOTES_RELEASE_DIR>


#### NomBank

We use NomBank to train a nominal SRL parser, since the annotations of NomBank are provided on top of the Penn TreeBank 
we have to preprocess both external resources together. 
_Note_: Verify that the Penn TreeBank directory contains the trees in MRG format.

> python ./nombank/preprocess_nombank2.py --nombank_dir <NOMBANK_DIR> --penntreebank_dir <PTB_DIR>


## Aligning QA-SRL with PropBank
This part is described in section 4.1 of our paper.
It consists of joint processing of two datasets, OntoNotes (annotated with PropBank SRL)
and QA-SRL-Bank (annotated with questions and answers) each with parsers of the other formalism.
- QA-SRL Bank is parsed with PropBank SRL parser, producing sentences with role-labeled spans.
Then, the annotated answers are aligned with the labelled spans, and the label (A0) is applied to the QA-SRL question-answer pair.
- OntoNotes gold arguments are re-labelled with QA-SRL parser's questions.
 The produced question is added to the gold argument and role label.

However, the SRL parser actually consists of two different models, one for Verbal SRL (publicly available) 
and one for Nominal SRL (which we trained using the same model architecture and hyper-params as the verbal one).
We run the verbal SRL parser on top of QA-SRL Bank, which only contains verbal predicates, 
and the nominal SRL parser on top of QA-Nom, which has only deverbal noun predicates.

Moreover, while detecting verbal predicates is rather easy with a simple PoS tagger,
detecting noun predicates is a more delicate task. 
For this purpose we train a nominal predicate classifier using nominal predicates in OntoNotes.
    
#### Training Nominal Predicate Classifier with OntoNotes 5 Data
The following command will use pre-processed ontonotes.[train|dev].jsonl files 
to train a nominal predicate identifier:
> python ./ontonotes/finetune_predicate_indicator.py --model_name bert-base-uncased

The trained model will be saved under: ./experiments/nom_pred

#### Running nominal predicate detector using the trained model.
The following will take in sentences and output the same sentences with their detected noun predicates.
 The input and output should both be .jsonl files (JSON-lines, JSON object per line)
 with a property named 'text'. The output file will contain all the fields of the input with 
 the added property: "target_indicator" that would contain a list of token indices that correspond to a predicate.
  
> python ./ontonotes/predict_predicate_indicators.py --in_path <IN_FILE.jsonl> --out_path <OUT_FILE.jsonl>


#### Training Nominal SRL Parser using NomBank Data
To train our nominal SRL parser we use (and adjust) the following [config file from AllenNLP](https://github.com/allenai/allennlp-models/blob/main/training_config/structured_prediction/bert_base_srl.jsonnet).


#### Predicting questions for OntoNotes
We predict questions for OntoNotes by using the question generation part of the [QA-SRL model of Fitzgerald et al.](https://github.com/nafitzgerald/nrl-qasrl).


#### Aligning Predicted SRL arguments with QA-SRL question answer pairs.
 - Aligning annotated QA-SRL question-answer pairs with predicted (verbal) SRL spans and labels
  > python align_QASRL.py

 - Aligning annotated QA-Nom question-answer pairs with predicated (nominal) SRL spans and labels
  > python align_QANOM.py


#### Merging and Unifying and further Processing QA-SRL datasets.
The previous step results in multiple intermediate results, we would like to unify these:
 - QA-SRL and QA-Nom that are aligned to a PropBank role and unified into a single file.
 - Merged with QA-SRL Frame-Aligned Bank
 - Decompose Questions to its prototype form
TODO @plroit 
 > python ./qasrl/unify_qasrl_qanom_roles_placeholder_fillers.py
 

## Training the Contextualizer


#### Preparing Training Data
This step creates source and target training files for the seq2seq model employed by the contextualizer.
> python ./bart_transformation/create_training_data_question_transf.py

#### Running the Training Process
We trained the contextualizer using Huggingface's [BART seq2seq implementation (legacy)](https://github.com/huggingface/transformers/tree/master/examples/legacy/seq2seq).

## Running Prototype Selector
This step may run for a long time, it is designed to run on parallel on 10 nodes
by supplying an index from 0 to 9 (10 is a hardcoded value). 
Then you can run this script with a special option to merge and process all outputs.
TODO @plroit 
> python ./prototypes/calc_prototype_accuracy.py --index 0
> python ./prototypes/calc_prototype_accuracy.py --index 1
> ... 
> python ./prototypes/calc_prototype_accuracy.py --index 9
> python ./prototypes/calc_prototype_accuracy.py --merge_all



