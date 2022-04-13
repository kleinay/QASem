import jsonlines
import codecs
import re
from collections import defaultdict

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


def check_for_alignment(qa_verb_index, srl_verb_index, srl_arg, qa_answer):
    if qa_verb_index == srl_verb_index:
        arg_iou = get_IOU(srl_arg, qa_answer)
        if arg_iou >= 0.4:
            return True
        else:
            return False

def align_with_qasrl(pred_dict, outfile_path, infile_path):
    infile = codecs.open(infile_path, 'r')
    outfile = codecs.open(outfile_path, 'w')
    outfile.write('Sentence\tQASRL_question\tQASRL_answer\tQASRL_answer_indices\tQASRL_id\tQASRL_predicate\tQASRL_predicate_index\tQASRL_predicate_stem\tmodel_target\tmodel_target_token\tmodel_argument\trole\tsrl\n')
    counter = 0
    for inline in infile.readlines():
        line = inline.split('\t')[:-1]
        sent = line[0]
        qa_verb = line[5]
        qa_verb_lemma = line[7]
        qa_question = line[1]
        qa_answers = line[2]
        qa_answer_idx = line[3]
        qa_verb_index = int(line[6].split(':')[0])
        qa_verb_index_orig = line[6]
        ID = line[4]
        if ID in pred_dict:
            predictions = pred_dict[ID]
            for prediction in predictions:
                target_verb_idx = int(prediction[-1][0])
                target_verb_word = prediction[-1][1]
                roles_out = [' '.join(role_prediction[0])+' '+role_prediction[1] for role_prediction in prediction[:-1]]
                for arg_inst in prediction[:-1]:
                    arg = ' '.join(arg_inst[0])
                    label = arg_inst[1]
                    alignment = check_for_alignment(qa_verb_index, target_verb_idx, arg, qa_answers)
                    if alignment:
                        outfile.write(
                            sent + '\t' + qa_question + '\t' + qa_answers + '\t' + qa_answer_idx + '\t' + ID + '\t' + qa_verb + '\t' + qa_verb_index_orig + '\t' + qa_verb_lemma + '\t' + str(target_verb_idx) + '\t' + target_verb_word  + '\t' + arg + '\t' + label + '\t'+ '$$$'.join(roles_out)+'\n')
                        counter += 1
    outfile.close()


def get_predicted(srl_predictions_file_path, predict_file_path):
    #ID : args_senses
    pred_dict = defaultdict(lambda: [])
    infile = jsonlines.open(srl_predictions_file_path)
    instance_infile = jsonlines.open(predict_file_path)
    counter = 0
    for obj, instance in zip(infile, instance_infile):
        if len(obj["verbs"])>0:
            for verb in obj["verbs"]:
                tags = verb["tags"]
                ID = instance["ID"]
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
    # path to the predictions made by the verbal SRL model
    srl_predictions_file_path = ''
    # path to the input file when doing the predictions with the verbal SRL model
    predict_file_path = ''
    # desired output path
    outfile_path = ''
    # path to QASRL data
    infile_path = ''
    pred_dict = get_predicted(srl_predictions_file_path, predict_file_path)
    align_with_qasrl(pred_dict, outfile_path, infile_path)