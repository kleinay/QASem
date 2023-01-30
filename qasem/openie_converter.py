from typing import Tuple, Iterable, List, Any, Dict
import json
import itertools 

OIE = Tuple[str, ...]
default_layers_included = ["qasrl"]

class OpenIEConverter:
    """
    Converts a set of QA-SRL QAs into Open Information Extraction tuples.
    The OpenIE tuples (=propositions) have >3 slots, where first is subject, 
    second is predicate, and the rest are further arguments. 
    """
    _all_wh_words = ("how long", "how much", "how", "when", "where", "why", "what", "who")
    _adjunt_wh_words = ("how long", "how much", "how", "when", "where", "why")
    def __init__(self, 
                 layers_included: List[str] = default_layers_included,
                 labeled_adjuncts: bool = False):
        self.layers_included = layers_included
        self.labeled_adjuncts = labeled_adjuncts
    
    @staticmethod
    def _get_question_wh(question: str) -> str:
        question= question.lower().strip()
        for wh in OpenIEConverter._all_wh_words:
            if question.startswith(wh):
                return wh
        
    def convert_qadiscourse_qas(self, qas) -> List[OIE]:
        """ Converts QADiscourse information into OpenIE propositions. """
        # Future: consider how to convert qadiscourse QAs into propositions
        # qas = self.filter_qadiscourse_redundant_QAs(qas)
        # return [proposition 
        #         for qa in qas 
        #         for proposition in self.convert_single_qadiscourse_qa(qa)]
        
        # Currently, don't translate QADiscourse QAs into propositions
        return []
        
    def convert_single_qadiscourse_qa(self, qa) -> OIE:
        """ Converts a single QADiscourse qa into a single OpenIE propositions. """
        raise NotImplementedError()
    
    def filter_qadiscourse_redundant_QAs(self, qas) -> List[Dict[str, str]]:
        # remove equivalent QAs
        return [dict(t) for t in {tuple(qa.items()) for qa in qas}]
    
    
    def convert_single_qasrl_predicate(self, pred_info, sentence) -> List[OIE]:
        """ Converts QASRL information of a verb or a nominalization into OpenIE propositions. """
        # collect arguments (grouped by question) along with their index and wh-word
        args = [[(ans, sentence.index(ans), self._get_question_wh(qa["question"])) for ans in qa["answers"]]
                 for qa in pred_info["QAs"]]
        # sort by first occurrence
        args = list(sorted(args, key=lambda answers: min(t[1] for t in answers)))
        # omit repetitions (full or partial)
        def get_answers_without_repetitions(answers: List[Tuple[str, int]]):
            answers_no_repetitions = answers.copy()
            i=1
            while i < len(answers_no_repetitions): 
                if answers_no_repetitions[i] in answers_no_repetitions[i-1]:
                    del answers_no_repetitions[i]
                else:
                    i += 1
            return answers_no_repetitions
        args = [get_answers_without_repetitions(answers) for answers in args]

            
        # omit indices
        def arg_to_str(arg: Tuple[str, int, str]) -> str:
            s = arg[0]
            if self.labeled_adjuncts and arg[2] in OpenIEConverter._adjunt_wh_words:
                s = f"{arg[2].title()}: {arg[0]}"
            return s
        args_strs = [[arg_to_str(a) for a in arg] for arg in args]
        # add predicate to args
        # args.insert(1, [(pred_info["predicate"], pred_info["predicate_idx"])])
        args_strs.insert(1, [pred_info["predicate"]])
        # yield Cartesian product of args
        propositions = []
        for tup in itertools.product(*args_strs):
            propositions.append(tup)
        return propositions
        
    
    def convert_single_sentence(self, sent_info, sentence: str) -> List[OIE]:
        propositions: List[OIE] = []
        # Collect from QASRL and QANom layers
        qasrl_propositions: List[OIE] = []
        qadisc_propositions: List[OIE] = []
        for layer in self.layers_included:
            if layer in ("qasrl", "qanom"):
                # iterate verbal/nominal predicates
                for pred_info in sent_info[layer]:
                    qasrl_propositions.extend(self.convert_single_qasrl_predicate(pred_info, sentence))
            if layer == "qadiscourse":
                qadisc_propositions.extend(self.convert_qadiscourse_qas(sent_info["qadiscourse"]))
        propositions = qasrl_propositions + qadisc_propositions
        return propositions
                
    
def test_openie_converter():
    converter = OpenIEConverter(labeled_adjuncts=True)
    input_sentences = ["The doctor was very interested in Luke 's treatment as he was not feeling well .",
                       "Tom brings the dog to the park."]
    pipe_output = json.load(open("tmp/example_output.json"))
    oie_outputs = [converter.convert_single_sentence(sent_info, sent)
                   for sent_info, sent in zip(pipe_output, input_sentences)]
    print(oie_outputs)
    

if __name__ == "__main__":
    test_openie_converter()