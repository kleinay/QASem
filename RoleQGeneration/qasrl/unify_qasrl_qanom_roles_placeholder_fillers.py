import pandas as pd
import spacy
from jsonlines import jsonlines
from tqdm import tqdm
from lemma_utils import parse_span, get_lemma

tqdm.pandas()

HAVE_AUX = {'has', 'had', 'hasn\'t', 'hadn\'t'}

VERB_LABELS = [
    'stem',
    'pastParticiple',
    'presentSingular3rd',
    'past',
    'being pastParticiple',
    'presentParticiple',
    'be pastParticiple',
    'been pastParticiple',
    'be presentParticiple',
    'have pastParticiple',
    'been presentParticiple',
    'have been pastParticiple',
    'not stem',
    'not be pastParticiple',
    'have been presentParticiple',
    'not be presentParticiple',
    'not have pastParticiple',
    'not have been pastParticiple',
    'not have been presentParticiple',
]


def is_aux_verb_passive(arg):
    # This has been found by inspecting QASRL-BANK.
    # QA-NOM was sourced with a slightly different procedure,
    # and passive arguments there don't necessarily coincide
    # with these rules.
    return ('pastParticiple' in arg['verb']) and arg['aux'] not in HAVE_AUX


def is_simple_question(q: str):
    for modal in ['might', "should", 'ca', 'can', 'could', 'would']:
        if modal in q:
            return False
    if "n't" in q:
        return False
    return True


def is_aux_negated(aux: str):
    return 'n\'t' in aux


def fix_role(role):
    return role.replace("C-", "").replace("R-", "").replace("ARG", "A")


def combine_qasrl_id(r):
    sep = "_" if r.doc_id.startswith("TQA") else ":"
    return f"{r.doc_id}{sep}{r.sent_id}"


def load_questions(qasrl_path, qasrl_role_aligned_path):
    # qasrl_df contains the question slots (wh, subj, obj, etc...)
    # while the role_aligned DF contains the parsed SRL roles.
    # Need both to get question prototypes for each role.
    qasrl_df = pd.read_csv(qasrl_path, sep="\t")
    if 'role_type' in qasrl_df.columns:
        qasrl_df.drop('role_type', axis="columns", inplace=True)
    role_df = pd.read_csv(qasrl_role_aligned_path, sep="\t",
                          usecols=['doc_id', 'predicate_span', 'question', 'role_type'])
    role_df.role_type = role_df.role_type.apply(fix_role)
    if 'sent_id' in qasrl_df.columns:
        qasrl_df['doc_id'] = qasrl_df.apply(combine_qasrl_id, axis="columns")
    qasrl_df = pd.merge(qasrl_df, role_df, on=['doc_id', 'predicate_span', 'question'], how="left")
    qasrl_df.role_type.fillna("UNK", inplace=True)
    qasrl_df['predicate_lemma'] = qasrl_df.progress_apply(lambda r: get_lemma(r.text.split(),
                                                                     parse_span(r.predicate_span)[0], nlp), axis='columns')
    # DO THE FIELDS IN QANOM (VERB, AUX) REALLY CORRESPOND TO THOSE IN QASRL?
    # AND DO THOSE FIELDS CORRESPOND TO WHAT IS PREDICTED BY THE QASRL PARSER?
    if 'verb_form' not in qasrl_df.columns:
        qasrl_df['verb_form'] = qasrl_df['predicate_lemma'].copy()
    for col in ['subj', 'obj', 'prep', 'aux', 'obj2']:
        qasrl_df[col].fillna("_", inplace=True)
    qasrl_df = qasrl_df[['predicate_lemma', 'role_type',
                         'question', 'wh', 'aux', 'subj',
                         'verb', 'obj', 'prep', 'obj2', 'is_passive',
                         'verb_form', 'doc_id', 'predicate_span', 'text']].copy()
    # Discard what cannot be saved
    qasrl_df = qasrl_df[qasrl_df.verb.isin(VERB_LABELS)].copy()
    return qasrl_df


def load_onto_args(onto_path):
    onto = list(jsonlines.open(onto_path))
    all_args = []
    for sent in tqdm(onto):
        # if len(all_args) > 2000:
        #     break
        tokens = sent['text'].split()
        pred_idx = parse_span(sent['predicate']['span'])[0]
        lemma = get_lemma(tokens, pred_idx, nlp)
        for arg in sent['arguments']:
            role = arg['role_type']
            # I DISLIKE ADJUNCTS, BUT WHAT TO DO
            if "ARGM" in role:
                continue

            new_arg = dict(arg)
            # remove argument specific data, leave only the predicate and role.
            del new_arg['head_idx']
            del new_arg['arg_type']
            del new_arg['text']
            del new_arg['span']
            new_arg['role_type'] = fix_role(role)
            is_passive = is_aux_verb_passive(arg)
            new_arg['is_passive'] = is_passive
            new_arg['predicate_lemma'] = lemma
            all_args.append(new_arg)
    arg_df = pd.DataFrame(all_args)
    arg_df['is_negated'] = arg_df.aux.apply(is_aux_negated)
    arg_df = arg_df[['predicate_lemma', 'role_type', 'question', 'wh', 'aux',  'subj',
                     'verb', 'obj', 'prep', 'obj2',
                     'is_negated', 'is_passive']].copy()
    return arg_df


def yield_placeholder_qasrl_questions(test_path):
    sents = list(jsonlines.open(test_path))
    for s in sents:
        doc_id = s['sentenceId']
        for verb in s['verbs'].values():
            pred_idx = verb['verbIndex']
            predicate_span = f"{pred_idx}:{pred_idx+1}"
            for question, question_entries in verb['questions'].items():
                filled_questions = question_entries.keys()
                filled_questions = "~!~".join(filled_questions)
                yield {
                    "doc_id": doc_id,
                    "predicate_span": predicate_span,
                    "question": question,
                    "filled_question": filled_questions
                }


if __name__ == "__main__":
    # nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # for part in ['dev', 'train']:
    #     qasrl_pb_aligned_path = f"../Data/RoleAlignedDatasets/qasrl_pb.orig.{part}.tsv"
    #     qasrl_path = f"../Data/qasrl/qasrl.expanded.{part}.tsv"
    #     qanom_nb_aligned_path = f"../Data/RoleAlignedDatasets/qanom_nb.orig.{part}.tsv"
    #     qanom_path = f"../Data/qanom/qanom.{part}.tsv"
    #     test_path = "../Data/qasrl/qasrl-question-alignments.jsonl"
    #     placeholder_samples = list(yield_placeholder_qasrl_questions(test_path))
    #     qasrl_placeholder = pd.DataFrame(placeholder_samples)
    #     qasrl_df = load_questions(qasrl_path, qasrl_pb_aligned_path)
    #     qanom_df = load_questions(qanom_path, qanom_nb_aligned_path)
    #     ques_df = pd.concat([qasrl_df, qanom_df]).dropna()
    #     ques_df = pd.merge(ques_df, qasrl_placeholder, on=['doc_id', 'predicate_span', 'question'], how="left")
    #     ques_df.to_csv(f"../Data/qasrl/qasrl_qanom.{part}.tsv", index=False, encoding="utf-8", sep="\t")

    # HACK FOR VALENTINA - FULL QASRL (not just role aligned) with Ju
    part = "dev"
    qasrl_contextualized_path = f"../Data/qasrl/qasrl_qanom.filled.corrected.{part}.tsv"
    qasrl_path = f"../Data/qasrl/qasrl.expanded.{part}.tsv"
    qanom_path = f"../Data/qanom/qanom.{part}.tsv"
    orig_dfs = []
    cols = ['doc_id', 'predicate_span', 'question', 'gold_answers', 'gold_answer_spans']
    for p in (qasrl_path, qanom_path):
        orig_df = pd.read_csv(p, sep="\t")
        if 'sent_id' in orig_df.columns:
            orig_df['doc_id'] = orig_df.apply(combine_qasrl_id, axis="columns")
        orig_df = orig_df[cols].copy()
        orig_dfs.append(orig_df)
    orig_df = pd.concat(orig_dfs)
    qasrl_df = pd.read_csv(qasrl_contextualized_path, sep="\t")
    print(qasrl_df.shape)
    qasrl_df = pd.merge(qasrl_df, orig_df, on=['doc_id', 'predicate_span', 'question'], how="left")
    print(qasrl_df.shape)
    print(qasrl_df.gold_answers.isnull().sum())
    qasrl_df.to_csv(qasrl_contextualized_path)
    # END HACK

    # cols = ['predicate_lemma', 'role_type', 'question', 'wh', 'aux',  'subj',
    #         'verb', 'obj', 'prep', 'obj2', 'is_passive']
    # onto_df = load_onto_args("../Data/ontonotes/ontonotes.qasrl_questions.jsonl")[cols].copy()
    # onto_df.to_csv("../Data/ontonotes/ontonotes.qasrl_questions.tsv", sep="\t", index=False, encoding="utf-8")



