from argparse import ArgumentParser
from glob import glob
from xml import etree

import pandas as pd
from tqdm import tqdm

from role_lexicon.preprocess_roles_base import yield_roles, yield_predicate_nodes, to_json


# ```xml
# <predicate lemma="ghost">
#     <roleset id="ghost.01" name="move like a ghost">
#       <aliases>
#         <alias framenet="" pos="v" verbnet="">ghost</alias>
#       </aliases>
#       <note>GHOST-V NOTES: Roleset by Katie based on internet search, parallel expansion of verbnet,...</note>
#       <roles>
#         <role descr="Moving entity" f="PPT" n="0">
#           <vnrole vncls="51.3.2" vntheta="Theme"/>
#         </role>
#         <role descr="path or location" f="LOC" n="1">
#           <vnrole vncls="51.3.2" vntheta="Location"/>
#         </role>
#       </roles>
#     </roleset>
# </predicate>
# ```
def yield_predicate_roles(predicate_paths):
    for pred_path, pred_node in yield_predicate_nodes(predicate_paths):
        roleset_nodes = pred_node.findall("roleset")
        for role_set_node in roleset_nodes:
            pos = "UNK"
            alias_nodes = role_set_node.findall("./aliases/alias")
            if alias_nodes:
                pos = alias_nodes[0].attrib['pos']

            roleset_id = role_set_node.attrib['id']
            pred_lemma, sense_id, sense_particle = get_predicate_lemma_and_sense(roleset_id)
            # Not interested in light verbs...
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


def get_predicate_lemma_and_sense(predicate_id):
    pred_lemma, sense_id = predicate_id.split(".")
    sense_particle = "_"
    if "_" in pred_lemma:
        pred_lemma, sense_particle = pred_lemma.split("_", maxsplit=1)
    return pred_lemma, sense_particle, sense_id


if __name__ == "__main__":
    ap = ArgumentParser()
    pb_root = "C:\\dev\\propbank-frames\\frames"
    predicate_paths = glob(f"{pb_root}/*.xml")
    all_roles = list(tqdm(yield_predicate_roles(predicate_paths)))
    df = pd.DataFrame(all_roles)
    df = df[['predicate_lemma', 'sense_id', 'pos', 'sense_particle', "role_type",
             "roleset_desc", 'role_desc', 'function_tag', "vn_theta"]].copy()
    df.sort_values(['predicate_lemma', 'sense_id', 'role_type'], inplace=True)
    df.to_csv("./role_lexicon/predicate_roles.unified_propbank.tsv", sep="\t", index=False)
    to_json(df, "./role_lexicon/frames.unified_propbank.jsonl")
