import itertools
import os
from argparse import ArgumentParser
from collections import defaultdict
from random import shuffle

import jsonlines
import pandas as pd
import spacy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from lemma_utils import get_lemma, get_lemma_pos

tqdm.pandas()
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

from question_translation import QuestionTranslator, PREDICATE_PLACEHOLDER
from qa_models import QuestionAnswerModule, QuestionAnswerDataset, calc_score, parse_span
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from .normalize_qasrl_questions import load_wiktionary, normalize_question

CORE_ROLES = [f"A{i}" for i in range(6)]
ADJ_ROLES = ["AM-LOC",  # Locative
             "AM-TMP",  # Temporal
             "AM-MNR",  # Manner
             "AM-CAU",  # Cause
             "AM-PRP",  # Purpose
             "AM-GOL",  # Goal
             "AM-EXT",  # Extent
             ]

# Predicates for 100 Annotated Questions from Du and Cardie, ACL 2020
CARDIE_PREDICATES = {'open', 'conquer', 'coup', 'acquire', 'create', 'convict',
                     'name', 'email', 'birth', 'capture',  'come', 'divorce', 'drop', 'confer',
                     'wound', 'appoint', 'charge', 'contribute', 'bring', 'talk',
                     'trial', 'cease', 'arrest', 'execute', 'release', 'speed',
                     'march', 'sentence', 'leave', 'select', 'step', 'pay', 'give', 'prosecute',
                     'bankrupt', 'head', 'take', 'wed', 'serve', 'rally', 'shoot', 'elect', 'extradite',
                     'pardon', 'kill', 'appeal', 'die', 'become', 'enter', 'donate', 'telephone', 'fine',
                     'sue', 'acquit', 'buy', 'combat', 'return'}

# ON5V: Moor, Roth, Frank 2013
MOOR_PREDICATES = {"bring", "pay", "leave", "put", "give"}

# G&C: Gerber and Chai 2010:
GC_PREDICATES = {'sell', 'plan', 'fund', 'price', 'cost', 'lose', 'invest', 'loan', 'bid'}

MAX_TEXT_LENGTH = 192
MAX_QUESTION_LENGTH = 30
BATCH_SIZE = 8
TOP_K_PROTOS = 10
SAMPLES_PER_ROLE_SENSE = 100

# DEBUG
# TOP_K_PROTOS = 1
# SAMPLES_PER_ROLE_SENSE = 1


def filter_core_or_adjuncts(df: pd.DataFrame, is_core_role: bool):
    if is_core_role:
        df = df[df.role_type.isin(CORE_ROLES)].copy()
    else:
        df = df[df.role_type.isin(ADJ_ROLES)].copy()
    return df


HAVE_AUX = {'has', 'had', 'hasn\'t', 'hadn\'t'}


def is_aux_verb_passive(arg):
    # This has been found by inspecting QASRL-BANK.
    # QA-NOM was sourced with a slightly different procedure,
    # and passive arguments there don't necessarily coincide
    # with these rules.
    return ('pastParticiple' in arg['verb']) and arg['aux'] not in HAVE_AUX


# Step 1: Collect all prototypes from QA-SRL
def step_1_collect_all_protos(qasrl_df, onto_df, wiktionary, is_core: bool):
    grouped_dfs = []
    group_cols = ['verb_form', 'role_type', 'proto_question']
    proto_top_k = TOP_K_PROTOS if is_core else TOP_K_PROTOS*10
    for df, prefix in [(qasrl_df, "qasrl"), (onto_df, "onto")]:
        # for adjuncts we ommit the exact verb-form, by using a placeholder value
        # we effectively group on the same verb-form for all adjuncts.
        if not is_core:
            df = df.copy()
            df['predicate_lemma'] = PREDICATE_PLACEHOLDER
            df['verb_form'] = PREDICATE_PLACEHOLDER
        df['proto_question'] = df.progress_apply(lambda s: normalize_question(s, wiktionary), axis="columns")
        df2 = df.groupby(group_cols).size().rename(f'proto_count').reset_index()
        # get top-10 from both sides
        df2.sort_values(['verb_form', 'role_type', 'proto_count'],
                        ascending=[True, True, False], inplace=True)
        df3 = df2.groupby(['verb_form', 'role_type']).head(proto_top_k)
        grouped_dfs.append(df3)
    proto_df = pd.merge(grouped_dfs[0], grouped_dfs[1], on=group_cols, how="outer", suffixes=['_qasrl', "_onto"])
    proto_df.proto_count_qasrl.fillna(0, inplace=True)
    proto_df.proto_count_onto.fillna(0, inplace=True)
    qasrl_groups = proto_df.groupby(['verb_form', 'role_type'])
    all_protos = {}
    for (verb_form, role_type), group_df in tqdm(qasrl_groups, desc="Grouping QASRL Prototypes"):
        role_samples = group_df.to_dict(orient="records")
        proto_questions = [s['proto_question'] for s in role_samples]
        all_protos[(verb_form, role_type)] = proto_questions

    return proto_df, all_protos


# Step 2: Load all OntoNotes arguments by predicate and role.
def step_2_load_ontonotes_arguments(ontonotes_path, is_core, my_predicates):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    onto = list(jsonlines.open(ontonotes_path))
    samples = []
    for sent in tqdm(onto, desc="Loading OntoNotes Arguments..."):
        tokens = sent['text'].split()
        for frame in sent['frames']:
            predicate = frame['predicate']
            pred_idx = parse_span(predicate['span'])[0]
            lemma = predicate.get('predicate_lemma')
            if my_predicates and lemma not in my_predicates:
                continue
            if not lemma:
                lemma = get_lemma(tokens, pred_idx, nlp)
            verb_forms, is_ok = get_verb_forms_from_lexical_resources(lemma)
            if not is_ok:
                continue
            verb_form = verb_forms[0]

            for arg in frame['arguments']:
                role = arg['role_type']
                if is_core and role not in CORE_ROLES:
                    continue
                # for adjuncts, skip "to be" or "to have" aux predicates.
                if not is_core and (role not in ADJ_ROLES or verb_form in ('be', 'have')):
                    continue
                if arg['arg_type'] != 'explicit':
                    continue
                arg['predicate_lemma'] = lemma
                arg['predicate_span'] = predicate['span']
                arg['verb_form'] = verb_form
                arg['gold_answer_spans'] = [parse_span(arg['span'])]
                arg['gold_answers'] = [arg['text']]
                arg['text'] = sent['text']
                arg['sense_id'] = predicate['frame_id']
                del arg['span']
                samples.append(arg)
    return samples


# Step 5: Run examples through SQuAD QA model.
def step_5_infer_answers_to_qas(samples, qa_module, qa_tokenizer, device):
    dataset = QuestionAnswerDataset(samples, qa_tokenizer,
                                    max_length=MAX_TEXT_LENGTH,
                                    use_predicate_indicator=False)
    data_loader = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             collate_fn=dataset.tokenize_collate_with_samples)
    returned_samples = []
    for batch_samples, batch in tqdm(data_loader, desc="Predicting Answers", leave=False):
        spans = qa_module.infer(batch_samples, batch, device)
        for sample, span in zip(batch_samples, spans):
            start, end = span
            sample['predicted_answer_span'] = span
            tokens = sample['text'].split()
            answer_text = " ".join(tokens[start: end])
            sample['predicted_answer'] = answer_text
            returned_samples.append(sample)
    return returned_samples


# Step 6: Collect predicted answer accuracy for each predicate-role-sense-prototypea
def step_6_calc_metrics_for_predicted_answers(samples):
    samples = sorted(samples, key=lambda s: s['proto_question'])
    question_scores = []
    for proto_question, samples_per_question in itertools.groupby(samples, key=lambda s: s['proto_question']):
        samples_per_question = list(samples_per_question)
        scores = [calc_score(s) for s in samples_per_question]
        squad_f1 = pd.DataFrame(scores).f1.mean()
        question_scores.append((proto_question, squad_f1))
    return question_scores


def step_3_combine_arguments_with_adjunct_prototypes(role2proto_questions, onto_samples):
    onto_df = pd.DataFrame(onto_samples)
    onto_df = onto_df.sample(frac=1.0)
    onto_df = onto_df.groupby("role_type").head(SAMPLES_PER_ROLE_SENSE)
    tq_iter = tqdm(onto_df.groupby("role_type"))
    for adj_role, onto_role_df in tq_iter:
        onto_role_samples = onto_role_df.to_dict(orient="records")
        proto_questions = role2proto_questions.get((PREDICATE_PLACEHOLDER, adj_role), [])
        if not proto_questions:
            continue
        tq_iter.set_postfix({"role": adj_role})
        cross_samples = []
        for proto_question, onto_sample in itertools.product(proto_questions, onto_role_samples):
            new_sample = dict(onto_sample)
            new_sample['proto_question'] = proto_question
            cross_samples.append(new_sample)
        yield PREDICATE_PLACEHOLDER, adj_role, "01", cross_samples


def step_3_combine_arguments_and_prototypes(role2proto_questions, onto_samples,
                                            all_roles,
                                            include_predicates=None):
    role2samples = defaultdict(list)
    for s in onto_samples:
        role2samples[(s['verb_form'], s['role_type'], s['sense_id'])].append(s)
    role2samples = dict(role2samples)

    roles_we_need = [role_entry for role_entry in all_roles
                     if not include_predicates or (role_entry['predicate_lemma'] in include_predicates)]

    tq_iter = tqdm(roles_we_need)
    for role_entry in tq_iter:
        verb_form, role_type, sense = role_entry['predicate_lemma'], role_entry['role_type'], role_entry['sense_id']
        tq_iter.set_postfix({"verb_form": verb_form, "sense": sense})

        sense = f"{sense:02d}"
        # Step 3: Combine each argument with each QASRL prototype
        proto_questions = role2proto_questions.get((verb_form, role_type), [])
        onto_samples = role2samples.get((verb_form, role_type, sense), [])
        # Take only upto 50 random samples of this predicate and role.
        # We need an approximate accuracy, not exact on this dataset.
        shuffle(onto_samples)
        onto_samples = onto_samples[:SAMPLES_PER_ROLE_SENSE]
        cross_samples = []
        for proto_question, onto_sample in itertools.product(proto_questions, onto_samples):
            new_sample = dict(onto_sample)
            new_sample['proto_question'] = proto_question
            cross_samples.append(new_sample)
        yield verb_form, role_type, sense, cross_samples


def step_4_predict_fillers_and_verb_inflections(samples, question_translator: QuestionTranslator):
    batch_size = BATCH_SIZE
    for batch_start in trange(0, len(samples), batch_size, desc="Translating Questions", leave=False):
        batch_end = batch_start + batch_size
        batch_samples = samples[batch_start: batch_end]
        questions = question_translator.predict(batch_samples)
        for s, q in zip(batch_samples, questions):
            s['question'] = q
    return samples


def get_my_predicates(roles, my_index):
    # my_predicates = GC_PREDICATES | MOOR_PREDICATES
    # return my_predicates
    # my_predicates = CARDIE_PREDICATES
    # my_verbs = []
    # for pred in my_predicates:
    #     verbs, is_ok = get_verb_forms_from_lexical_resources(pred)
    #     my_verbs.append(verbs[0])
    # return set(my_verbs)
    all_predicates = sorted(set([r['predicate_lemma'] for r in roles]))
    if my_index == -1:
        return all_predicates
    n_predicates = len(all_predicates)
    part_size = int(n_predicates / 10)
    my_start = my_index*part_size
    my_end = my_start + part_size
    print(my_index, my_start, my_end)
    my_predicates = all_predicates[my_start: my_end]
    return my_predicates


def load_onto_as_qasrl_questions(onto_qasrl_path, my_predicates):
    onto_frames = list(jsonlines.open(onto_qasrl_path))
    samples = []
    for frame in tqdm(onto_frames):
        predicate = frame['predicate']
        pred_span = predicate['span']
        pred_idx = parse_span(pred_span)[0]
        tokens = frame['text'].split()
        pred_lemma = predicate.get('predicate_lemma')
        pos = predicate.get("pos")
        if not pred_lemma or not pos:
            pred_lemma, pos = get_lemma_pos(tokens, pred_idx)
        if my_predicates and pred_lemma not in my_predicates:
            continue
        if not pos.upper().startswith("V"):
            continue
        verb_forms, is_ok = get_verb_forms_from_lexical_resources(pred_lemma)
        if not is_ok:
            continue
        verb_form = verb_forms[0]

        for arg in frame['arguments']:
            if arg['role_type'].startswith("R-") or arg['role_type'].startswith("C-"):
                continue
            sample = dict(arg)
            # keep only the question and the role fields
            del sample['span']
            del sample['text']
            del sample['arg_type']
            del sample['head_idx']
            sample['is_passive'] = is_aux_verb_passive(sample)
            sample['verb_form'] = verb_form.lower()
            sample['predicate_lemma'] = pred_lemma.lower()
            sample['role_type'] = sample['role_type'].replace("ARG", "A")
            samples.append(sample)
    return pd.DataFrame(samples)


def load_jsonl_roles(role_path):
    data = []
    for pred_entry in jsonlines.open(role_path):
        pred_lemma = pred_entry['predicate']
        for rs_entry in pred_entry['role_sets']:
            sense = rs_entry['sense_id']
            for role_entry in rs_entry['roles']:
                role_type = role_entry['type']
                role_desc = role_entry['desc']
                data.append({
                    "predicate_lemma": pred_lemma,
                    "sense_id": sense,
                    "role_type": role_type,
                    "role_desc": role_desc
                })
    return pd.DataFrame(data).drop_duplicates()


def main():
    qasrl_train_path = "../Data/qasrl/qasrl_qanom.train.tsv"
    ontonotes_train_path = "../Data/ontonotes/ontonotes.train.jsonl"
    # This should be only train samples.
    onto_qasrl_train_path = "../Data/ontonotes/ontonotes.qasrl_questions.jsonl"
    wiktionary_path = "../Data/en_verb_inflections.txt"
    # This is Unified PropBank (2016)
    # role_path = "../Data/predicate_roles.tsv"

    # This is OntoNotes 5 (2012)
    role_path = "../Data/frames.jsonl"

    # backward compatible roles for GC and MOOR.
    # GC roles are from the original NomBank (2010)
    # Moor roles are from OntoNotes 5
    # role_path = "../Data/predicate_roles.gc_moor.tsv"

    # This is the first translation model
    # trans_model_path = "../pretrained/question_transformation"

    # Grammar (is/are was/were) corrected and animacy stripped Who==>What
    trans_model_path = os.path.expanduser("~/pretrained/question_transformation_grammar_corrected")

    qa_model_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    # null is an actual verb! haha! set keep-default-na = False
    # core_roles_df = pd.read_csv(role_path, sep="\t", keep_default_na=False).to_dict(orient="records")
    core_roles_df = load_jsonl_roles(role_path).to_dict(orient="records")
    adj_roles_df = pd.DataFrame({
        "verb_form": [PREDICATE_PLACEHOLDER for _ in ADJ_ROLES],
        "role_type": [adj for adj in ADJ_ROLES]
    })
    ap = ArgumentParser()
    ap.add_argument("--gen_device", type=int, default=-1)
    ap.add_argument("--qa_device", type=int, default=-1)
    ap.add_argument("--my_index", type=int, default=-1)
    ap.add_argument("--is_core", type=int, default=1)
    args = ap.parse_args()
    is_core = bool(args.is_core)
    my_index = args.my_index
    my_predicates = get_my_predicates(core_roles_df, my_index)
    # result_path = f"../Data/qasrl/qasrl.prototype_accuracy.ontonotes_100.part_{my_index}.tsv"
    # result_path = f"../Data/qasrl/qasrl.prototype_accuracy.gc_moor.tsv"
    # result_path = f"../Data/qasrl/qasrl.prototype_accuracy.part_{my_index}.tsv"
    result_path = "../Data/qasrl/qasrl.prototype_accuracy.adjuncts.tsv"
    wiktionary = load_wiktionary(wiktionary_path)
    onto_df = load_onto_as_qasrl_questions(onto_qasrl_train_path, my_predicates)
    onto_df = filter_core_or_adjuncts(onto_df, is_core)
    qasrl_df = pd.read_csv(qasrl_train_path, sep="\t")
    qasrl_df = filter_core_or_adjuncts(qasrl_df, is_core)
    proto_df, role2proto = step_1_collect_all_protos(qasrl_df, onto_df, wiktionary, is_core)
    onto_samples = step_2_load_ontonotes_arguments(ontonotes_train_path, is_core, my_predicates)
    gen_device, qa_device = torch.device("cpu"), torch.device("cpu")
    if torch.cuda.is_available() and args.gen_device >= 0 and args.qa_device >= 0:
        qa_device = torch.device(f"cuda:{args.qa_device}")
    question_translator = QuestionTranslator.from_pretrained(trans_model_path, args.gen_device)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path).to(qa_device)
    qa_module = QuestionAnswerModule(qa_model)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path, use_fast=True)

    role_acc_results = []
    the_role_df = core_roles_df if args.is_core else adj_roles_df
    if args.is_core:
        tq_iter = step_3_combine_arguments_and_prototypes(
            role2proto, onto_samples,
            the_role_df, my_predicates)
    else:
        tq_iter = step_3_combine_arguments_with_adjunct_prototypes(role2proto, onto_samples)
    for res in tq_iter:
        verb_form, role_type, sense, cross_samples = res
        default_res = {"verb_form": verb_form,
                       "role_type": role_type,
                       "sense_id": sense,
                       "squad_f1": 0.0,
                       "support": 0,
                       'proto_question': "NO_QUESTION"}
        if not cross_samples:
            role_acc_results.append(default_res)
            continue
        cross_samples = step_4_predict_fillers_and_verb_inflections(cross_samples, question_translator)
        cross_samples = step_5_infer_answers_to_qas(cross_samples,
                                                    qa_module,
                                                    qa_tokenizer,
                                                    qa_device)

        question_accuracies = step_6_calc_metrics_for_predicted_answers(cross_samples)
        n_samples_in_sense = int(len(cross_samples)/len(question_accuracies))
        for proto_question, squad_f1 in question_accuracies:
            res = {"verb_form": verb_form,
                   "role_type": role_type,
                   "sense_id": sense,
                   "squad_f1": squad_f1,
                   "proto_question": proto_question,
                   "support": n_samples_in_sense}
            role_acc_results.append(res)
    cols = ['verb_form', 'sense_id', 'role_type', 'proto_question', 'support', 'squad_f1']
    df = pd.DataFrame(role_acc_results)[cols].copy()
    count_cols = ['verb_form', 'role_type', 'proto_question', 'proto_count_qasrl', 'proto_count_onto']
    df = pd.merge(df, proto_df[count_cols], on=['verb_form', 'role_type', 'proto_question'])
    df = df.sort_values(['verb_form', 'sense_id', 'role_type', 'squad_f1'], ascending=[True, True, True, False])
    df.to_csv(result_path, index=False, sep="\t", encoding="utf-8")


def combine_all_parts():
    dfs = [pd.read_csv(f"../Data/qasrl/qasrl.prototype_accuracy.ontonotes_100.part_{part_idx}.tsv",
                       sep="\t") for part_idx in range(10)]
    # dfs = [df[df.squad_f1 > 0.5].copy() for df in dfs]
    df = pd.concat(dfs)
    df = df.sort_values(['verb_form', 'sense_id', 'role_type', 'squad_f1'], ascending=[True, True, True, False])
    df2 = df.groupby(['verb_form', 'sense_id', 'role_type']).head(2)
    df2.to_csv(f"../Data/qasrl/qasrl.prototype_accuracy.ontonotes_100.tsv", sep="\t", index=False, encoding='utf-8')

    df3 = pd.read_csv(f"../Data/qasrl/qasrl.prototype_accuracy.ontonotes.tsv", sep="\t")
    cols = ['verb_form', 'sense_id', 'role_type']
    df2 = df2.groupby(cols).head(1)
    df3 = df3.groupby(cols).head(1)
    df4 = pd.merge(df2[cols+['proto_question']], df3[cols+['proto_question']], on=cols,
                   suffixes=['_100', "_50"])
    df4['is_diff'] = df4.proto_question_100 != df4.proto_question_50
    df4.to_csv("../Data/qasrl/qasrl.prototype_accuracy.ontonotes_100_vs_50.tsv", sep="\t", index=False)


def add_lemmas(qasrl_path):
    frames = list(jsonlines.open(qasrl_path))
    for frame in tqdm(frames, "Adding lemmas and PoS"):
        predicate = frame['predicate']
        pred_idx, _ = parse_span(predicate['span'])
        tokens = frame['text'].split()
        pred_lemma, pos = get_lemma_pos(tokens, pred_idx)
        predicate['pos'] = pos
        predicate['predicate_lemma'] = pred_lemma
    with jsonlines.open(qasrl_path, "w") as file_out:
        file_out.write_all(frames)


if __name__ == "__main__":
    main()
    # combine_all_parts()
    # onto_qasrl_train_path = "../Data/ontonotes/ontonotes.qasrl_questions.jsonl"
    # add_lemmas(onto_qasrl_train_path)



