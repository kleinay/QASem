from glob import glob

import pandas as pd
from tqdm import tqdm

from role_lexicon.preprocess_roles_base import yield_roles, yield_predicate_nodes, to_json


#     <predicate lemma="go_on">
#         <roleset framnet="Speak_on_topic" id="go.25" name="Continue speaking" vncls="-">
#             <roles>
#                 <role descr="speaker" f="" n="0"/>
#                 <role descr="utterance" f="" n="1"/>
#                 <role descr="listener, audience spoken to" f="" n="2"/>
#                 <note/>
#             </roles>
def yield_predicate_roles(predicate_paths):
    for pred_path, pred_node in yield_predicate_nodes(predicate_paths):
        # tackle-v.xml, charge-n.xml
        roleset_nodes = pred_node.findall("roleset")
        pos = pred_path.rsplit("-", maxsplit=1)[-1]
        pred_lemma = pred_node.attrib['lemma']
        sense_particle = "_"
        if "_" in pred_lemma:
            pred_lemma, sense_particle = pred_lemma.rsplit("_", maxsplit=1)
        for role_set_node in roleset_nodes:
            # "go.15"
            sense_id = role_set_node.attrib['id'].split(".")[-1]
            # Light verbs..
            if sense_id.upper() == "LV":
                continue
            for role in yield_roles(role_set_node):
                role.update({
                    "predicate_lemma": pred_lemma,
                    "sense_id": sense_id,
                    "sense_particle": sense_particle,
                    'pos': pos
                })
                yield role


if __name__ == "__main__":
    onto_root = "C:\\dev\\ontonotes\\ontonotes-release-5.0\\data\\files\\data\\english\\metadata\\frames"
    predicate_paths = glob(f"{onto_root}/*.xml")
    all_roles = list(tqdm(yield_predicate_roles(predicate_paths)))
    df = pd.DataFrame(all_roles)
    df.sort_values(['predicate_lemma', 'sense_id', 'role_type'], inplace=True)
    df = df[['predicate_lemma', 'sense_id', 'pos', 'sense_particle', "role_type",
             "roleset_desc", 'role_desc', 'function_tag', "vn_theta"]].copy()
    df.to_csv("./role_lexicon/predicate_roles.ontonotes.tsv", sep="\t", index=False)
    to_json(df, "./role_lexicon/frames.ontonotes.jsonl")
