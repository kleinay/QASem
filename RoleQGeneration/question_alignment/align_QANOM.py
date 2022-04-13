import codecs
import jsonlines
from collections import defaultdict
import spacy
import csv
import re
nlp = spacy.load('en_core_web_sm')


def get_IOU(text1, text2):
    iou = 0
    for answer in text2.split('~!~'):
        t1 = set(text1.lower().split())
        t2 = set(answer.lower().split())
        intersection = t1.intersection(t2)
        union = t1.union(t2)
        if len(intersection) / len(union) > iou:
            iou = len(intersection) / len(union)
    return iou

def check_for_alignment(qa_index, target_index, arg, answer):
    # if found ok, else: check if found when lemmatized
    if qa_index == target_index:
        arg_iou = get_IOU(arg, answer)
        if arg_iou >= 0.4:
            return True
        else:
            return False
    else:
        return False

def align_with_qanom(pred_dict, outfile_path, files):
    outfile = codecs.open(outfile_path, 'w')
    outfile.write('text\tquestion\tgold_answers\tgold_answer_spans\tdoc_id\tpredicate\tpredicate_span\tpredicate_lemma\tverbal_form\tmodel_target\tmodel_target_idx\tmodel_argument\trole_type\tsrl\n')
    counter = 0
    for file in files:
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sentence = row["sentence"]
                question = row["question"]
                answer = row["answer"]
                answer_indices = row["answer_range"]
                ID = row["qasrl_id"]
                qa_target_index = int(row["target_idx"])
                target_index_out_format = str(qa_target_index)+':'+str(qa_target_index+1)
                noun = row["noun"]
                verb = row["verb_form"]
                noun_doc = nlp(noun)
                noun_lemma = [token.lemma_ for token in noun_doc][0]
                if ID in pred_dict:
                    predictions = pred_dict[ID]
                    for prediction in predictions:
                        target_noun_idx = int(prediction[-1][0])
                        target_noun_word = prediction[-1][1]
                        roles_out = [' '.join(role_prediction[0]) + ' ' + role_prediction[1] for role_prediction in
                                     prediction[:-1]]
                        for arg_inst in prediction[:-1]:
                            arg = ' '.join(arg_inst[0])
                            label = arg_inst[1]
                            alignment = check_for_alignment(qa_target_index, target_noun_idx, arg, answer)
                            if alignment:
                                outfile.write(sentence+'\t'+question+'\t'+answer+'\t'+answer_indices+'\t'+ID+'\t'+noun+'\t'+target_index_out_format+'\t'+noun_lemma+'\t'+verb+'\t'+target_noun_word+'\t'+str(target_noun_idx)+'\t'+arg+'\t'+label+ '\t'+ '$$$'.join(roles_out)+'\n')
                                counter += 1
    outfile.close()

def get_predicted(srl_predictions_file_path):
    #ID : args_senses
    pred_dict = defaultdict(lambda: [])
    infile = jsonlines.open(srl_predictions_file_path)
    counter = 0
    for obj in infile:
        for verb in obj["verbs"]:
            tags = verb["tags"]
            ID = obj["verbs"][0]["ID"]
            if tags.count('O')>=len(tags)-1:
                pass
            else:
                words = obj["words"]
                target = []
                args_senses = []
                sense = ''
                arg = []
                counter = 0
                for tag, word in zip(tags, words):
                    if tag == 'O':
                        if len(arg)>0:
                            args_senses.append([arg, sense])
                        sense = ''
                        arg = []
                    elif "-V" in tag:
                        if len(arg)>0:
                            args_senses.append([arg, sense])
                        sense = ''
                        arg = []
                        target.append(str(counter))
                        target.append(word)
                    elif tag.startswith('B'):
                        if len(arg)>0:
                            args_senses.append([arg, sense])
                        arg = []
                        sense = '-'.join(tag.split('-')[1:])
                        sense = re.sub('ARG', 'A', sense)
                        arg.append(word)
                    else:
                        new_sense = '-'.join(tag.split('-')[1:])
                        new_sense = re.sub('ARG', 'A', new_sense)
                        if new_sense == sense:
                            arg.append(word)
                    counter += 1
                found = False
                for entry in pred_dict[ID]:
                    if entry[-1] == target:
                        found = True
                if not found and len(target)>0:
                    args_senses.append(target)
                    pred_dict[ID].append(args_senses)
                counter += 1
    return pred_dict

if __name__ == "__main__":
    # path to the predictions made by the nominal SRL model
    srl_predictions_file_path = ''
    # desired output path
    outfile_path = ''
    # path to QASRL files
    files = []
    pred_dict = get_predicted(srl_predictions_file_path)
    align_with_qanom(pred_dict, outfile_path, files)