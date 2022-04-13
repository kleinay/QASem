from argparse import ArgumentParser

from flask import Flask, jsonify, send_from_directory, request
import os
import spacy
import logging


if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

from demo.roleq_impl import setup_roleqs

WWWROOT = "wwwroot"
# logging.basicConfig(filename='roleqs.log', level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')


def setup():
    the_app = Flask(__name__, static_folder=WWWROOT)
    the_app.nlp = spacy.load("en_core_web_sm")
    os.environ['FLASK_ENV'] = "development"
    the_app.logger.info(f"Current Working Directory: {os.getcwd()}")
    the_app.logger.info(f"Root Path: {the_app.root_path}")
    the_app.logger.info(f"Root Path: {the_app.root_path}")
    return the_app


app = setup()


@app.route("/")
def index():
    return get_static_file("demo.html")


@app.route("/<path:static_file_path>")
def get_static_file(static_file_path):
    return send_from_directory(WWWROOT, static_file_path)


@app.route("/api/text", methods=["POST"])
def analyze():
    text = request.json['text']
    res = app.roleq.analyze(text)
    return jsonify(res)


@app.route("/api/rolesets", methods=["POST"])
def get_rolesets():
    req = request.json
    tokens = req["tokens"]
    selected_idx = req['predicate_idx']
    res = app.roleq.get_rolesets(selected_idx, tokens)
    return jsonify(res)


@app.route("/api/questions", methods=["POST"])
def get_questions():
    req = request.json
    lemma = req['lemma']
    pos = req['pos']
    sense_id = req['sense_id']
    tokens = req['tokens']
    predicate_idx = req['predicate_idx']
    res = app.roleq.get_questions(lemma, pos, sense_id, predicate_idx, tokens)
    return jsonify(res)


@app.route("/api/contextualize", methods=["POST"])
def contextualize():
    req = request.json
    pred_idx = req['predicate_idx']
    tokens = req['tokens']
    prototype = req['prototype']
    lemma = req['lemma']
    res = app.roleq.generate(prototype, pred_idx, tokens, lemma)
    return jsonify(res)


if __name__ == '__main__':
    pretrained_path = os.path.expanduser(os.path.expandvars("${PRETRAINED}"))
    default_model_path = os.path.join(pretrained_path, "question_transformation_grammar_corrected_who")
    ap = ArgumentParser()
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--trans_model", default=default_model_path)
    ap.add_argument("--proto_path", default="./resources/qasrl.prototype_accuracy.ontonotes.tsv")
    ap.add_argument("--lex_path", default="./role_lexicon/predicate_roles.ontonotes.tsv")

    args = ap.parse_args()
    role_demo = setup_roleqs(args)
    app.roleq = role_demo
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
