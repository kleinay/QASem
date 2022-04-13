from glob import glob
from itertools import groupby

import spacy
from jsonlines import jsonlines
from lxml import etree
import os

nlp = spacy.load('en_core_web_sm', disable=["ner", 'tagger', 'parser'])


def yield_predicate_nodes(predicate_paths):
    for pred_path in predicate_paths:
        # tackle-v.xml, charge-n.xml for ontonotes or tackle.xml
        pred_doc = etree.parse(pred_path)
        pred_root: etree.Element = pred_doc.getroot()
        predicate_nodes = pred_root.findall("predicate")
        for pred_node in predicate_nodes:
            file_name = os.path.splitext(os.path.basename(pred_path))[0]
            yield file_name, pred_node


def yield_roles(roleset_node):
    roleset_desc = roleset_node.attrib['name']
    role_nodes = roleset_node.findall("roles/role")
    for role_node in role_nodes:
        role_type = f"A{role_node.attrib['n'].upper()}"
        f_type = "_"
        if 'f' in role_node.attrib:
            f_type = role_node.attrib["f"].upper()
        desc = role_node.attrib['descr']
        vn_node = role_node.find("vnrole")
        vn_theta = "_"
        if vn_node is not None:
            vn_theta = vn_node.attrib['vntheta'].upper()
        yield {
            "roleset_desc": roleset_desc,
            "role_type": role_type,
            "role_desc": desc,
            "function_tag": f_type,
            "vn_theta": vn_theta
        }


def sense_pos_particle(t):
    return t['sense_id'], t['pos'], t['sense_particle']


def get_rolesets(datas):
    role_sets = []
    for (sense_id, pos, particle), roleset_group in groupby(datas, sense_pos_particle):
        roleset_entry = {
            "sense_id": sense_id,
            "sense_particle": particle,
            "pos": pos,
            "roles": []
        }
        for role_entry in roleset_group:
            roleset_entry['roles'].append({
                "type": role_entry['role_type'],
                "desc": role_entry['role_desc']
            })
        role_sets.append(roleset_entry)
    return role_sets


def to_json(df, out_path):
    # {"predicate": "go", "pos": "v", "role_sets": [{"sense_id": 1, "roles": [{"type": "A1", "desc": "entity in motion/goer"}
    df.sort_values(["predicate_lemma", "sense_id", "role_type"], inplace=True)
    datas = df.to_dict(orient="records")
    new_datas = []
    for predicate_lemma, predicate_group in groupby(datas, lambda t: t['predicate_lemma']):
        pred_entry = {
            "predicate": predicate_lemma,
            "role_sets": get_rolesets(predicate_group)
        }
        new_datas.append(pred_entry)
    with jsonlines.open(out_path, "w") as file_out:
        file_out.write_all(new_datas)


