from argparse import ArgumentParser

import pandas as pd
import json
import os

if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

from qa_models import parse_span


def parse_hit(state):
    datas = []
    prd_idx = state['predicate_idx']
    tokens = state['tokens']
    for item in state['roleset']:
        answer = ""
        answer_span = item['answer_span']
        start, end = parse_span(answer_span)
        if start != -1 and end != -1:
            answer = " ".join(tokens[start: end])

        data = {'role': item['role'],
                'question': item['question'],
                "answer": answer,
                'selected_descriptions': "~!~".join(item['descriptions']),
                "adequacy": item['adequacy'],
                "answer_span": item['answer_span'],
                "fluency": item['fluency'],
                "doc_id": state['doc_id'],
                "sent_id": state['sent_id'],
                "experiment_type": state['experiment_type'],
                "predicate_span": f"{prd_idx}:{prd_idx + 1}"}
        datas.append(data)
    return datas


def main(args):
    results = pd.read_csv(args.i, usecols=['Answer.results', "WorkerId"]).to_dict(orient='records')
    all_dfs = []
    for s in results:
        state = json.loads(s['Answer.results'])
        single_hit_results = parse_hit(state)
        df = pd.DataFrame(single_hit_results)
        df['worker_id'] = s["WorkerId"]
        all_dfs.append(df)
    df = pd.concat(all_dfs)
    df.to_csv(args.o, index=False, encoding="utf-8", sep="\t")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-i", help="/path/to/mturk_file.csv")
    ap.add_argument("-o", help="/path/to/results.csv")
    main(ap.parse_args())
