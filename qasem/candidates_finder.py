from typing import Optional 
import spacy
import codecs
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import defaultdict
from qasem import data_dir

connective_file_default_path = data_dir / "connectives_small_set.txt"

def get_connectives(conn_file):
    conns = defaultdict(lambda : '')
    infile = codecs.open(conn_file, 'r')
    for line in infile.readlines():
        line = line.split(',')
        conns[line[0]] = 'ok'
    return conns

def splitter(splittoken, mylist, post = True, add_white_sp = False):
    splitlist = []
    for span in mylist:
        if splittoken in span:
            newsplit = span.split(splittoken)
            newsplitlist = []
            if add_white_sp:
                if post:
                    prev_item_emtpy = False
                    for count, item in enumerate(newsplit):
                        if item.strip() == '':
                            prev_item_emtpy = True
                        elif prev_item_emtpy:
                            newsplitlist.append(item.strip() + ' ' + splittoken.strip())
                            prev_item_emtpy = False
                        elif count == len(newsplit)-1:
                            newsplitlist.append(item.strip())
                        else:
                            prev_item_emtpy = False
                            newsplitlist.append(item.strip()+' '+splittoken.strip())
                else:
                    for count, item in enumerate(newsplit):
                        if count == 0:
                            newsplitlist.append(item.strip())
                        else:
                            newsplitlist.append(splittoken.strip()+' '+item.strip())
            else:
                if post:
                    prev_item_emtpy = False
                    for count, item in enumerate(newsplit):
                        if item.strip() == '':
                            prev_item_emtpy = True
                        elif prev_item_emtpy:
                            newsplitlist.append(item.strip() + splittoken.strip())
                            prev_item_emtpy = False
                        elif count == len(newsplit) - 1:
                            newsplitlist.append(item.strip())
                        else:
                            prev_item_emtpy = False
                            newsplitlist.append(item.strip()  + splittoken.strip())
                else:
                    for count, item in enumerate(newsplit):
                        if count == 0:
                            newsplitlist.append(item.strip())
                        else:
                            newsplitlist.append(splittoken.strip() + item.strip())
            splitlist += newsplitlist
        else:
            splitlist.append(span)
    return splitlist
    
    
    
    
class CandidateFinder():

    def __init__(self, conn_file: Optional[str] = None):
        conn_file = conn_file or connective_file_default_path
        self.connectives = get_connectives(conn_file)
        self.detokenizer = TreebankWordDetokenizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.covered = defaultdict(lambda : '')
        
        
    def num_candidates(self, sentence):
        sentence = sentence.strip().split('\t')
        text = sentence[0]
        text = text.replace(' -LRB- ', ' ')
        text = text.replace(' -RRB- ', ' ')
        text = text.replace(' -LSB- ', ' ')
        text = text.replace(' -RSB- ', ' ')
        text = self.detokenizer.detokenize(text.split(' '))
        text = text.strip()
        text = text.replace('"', ' "" ')
        text = text.replace("''", ' "" ')
        text = text.replace("``", ' "" ')
        text = text.replace("'", "\\'")
        text = text.replace(" , ", ", ")
        # if text not in self.covered:
        #    self.covered[text] = True
        punc_split = splitter(', ', [text])
        punc_split = splitter('," ', punc_split)
        punc_split = splitter('; ', punc_split)
        punc_split = splitter(': ', punc_split)
        for conn in self.connectives.keys():
            if ' '+conn+' ' in text:
                punc_split = splitter(' '+conn+' ' , punc_split, post=False, add_white_sp = True)
        doc = self.nlp(text)
        word_to_pos = defaultdict(lambda : '')
        for count, token in enumerate(doc):
            word_to_pos[token.text] = token.tag_
        ids = []
        # Split if more than 1 verb:
        # VBD / VBN / VBP
        for spannum, span in enumerate(punc_split):
            verb_collector = []
            inner_verbs = []
            verb_counter = 0
            noun_collector = []
            other_collector = []
            prev_word_verb = False
            for wordnum, word in enumerate(span.split(' ')):
                word = word.strip()
                id = 0
                for i in range(spannum):
                    id += len(punc_split[i].split(' '))
                id += wordnum
                if word_to_pos[word] == '' and len(word) > 0:
                    if word[-2:].strip() in ['."', ',"']:
                        word = word[:-2]
                    elif word[-1] in [',', ';', ':', '.', '"']:
                        word = word[:-1]
                    elif word[-2:] in ["'d", "'s"]:
                        word = word[-2:]
                    elif word[-3:] in ["'re"]:
                        word = word[-3:]
                    elif word[-3:] in ["n't"] and word != "won't":
                        word = word[:-3]
                if word_to_pos[word].startswith('N'):
                    noun_collector.append(id)
                    prev_word_verb = False
                elif word_to_pos[word].startswith('RB'):
                    prev_word_verb = True
                elif word_to_pos[word].startswith('V'):
                    if word in ['said', 'according', 'spoke']:
                        pass
                    elif not prev_word_verb:
                        if len(inner_verbs)>0:
                            verb_collector.append(inner_verbs)
                        inner_verbs = []
                        inner_verbs.append(id)
                    else:
                        inner_verbs.append(id)
                    verb_counter += 1
                    prev_word_verb = True
                elif word == "n't":
                    prev_word_verb = True
                else:
                    other_collector.append(id)
                    prev_word_verb = False
            verb_collector.append(inner_verbs)
            if verb_counter == 0 and len(noun_collector)>0:
                if span.split(' ')[0].lower() in self.connectives and span.split(' ')[0].lower() != 'and':
                    ids.append(noun_collector[0])
            elif verb_counter == 0 and len(noun_collector) == 0:
                if span.split(' ')[0].lower() in self.connectives:
                    try:
                        ids.append(other_collector[1])
                    except IndexError:
                        pass
            elif verb_counter > 0:
                for v in verb_collector:
                    if len(v)>0:
                        ids.append(v[-1])
        return len(ids)

        
        