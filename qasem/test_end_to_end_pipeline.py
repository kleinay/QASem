from qasem.end_to_end_pipeline import QASemEndToEndPipeline
import spacy
from pprint import pprint
from tqdm import tqdm

if __name__ == "__main__":
    layers = [
        "qasrl",
        "qanom",
        "qadiscourse",
    ]
    open_ie_kwargs = {
        "layers_included": ["qasrl", "qanom"],
        "labeled_adjuncts": True,
    }
    pipe = QASemEndToEndPipeline(layers, 
                                 device=-1,
                                 nominalization_detection_threshold=0.75,
                                 contextualize=True,
                                 openie_converter_kwargs=open_ie_kwargs)
    sentences = ["The models went through various mutations"]
    sentences = ["The doctor was very interested in Luke 's treatment as he was not feeling well .", "Tom brings the dog to the park.", "I hate cats."]
    outputs = pipe(sentences, verbose=True, output_openie=True)
    pprint(outputs)
    # print()
    # sentences = ["the ball is red ."]
    # sentences = ["the construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."]
    # sentences = ["The Veterinary student was interested in Luke 's treatment of sea animals .", "John ate an apple because he was hungry ."]
    # outputs = pipe(sentences)
    # pprint(outputs)

    # pipe = QASemEndToEndPipeline(['qasrl'], nominalization_detection_threshold=0.75, contextualize = False)
    # sentences = ["The doctor was interested in Luke 's treatment .", "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ." , "The Veterinary student was interested in Luke 's treatment of sea animals .", "Tom brings the dog to the park."]
    # outputs = pipe(sentences)
    # print(outputs)
    # print()
    #
    # pipe = QASemEndToEndPipeline(['qadiscourse'])
    # sentence = "The construction of the officer 's building will finish after the rain would stop ."
    # print(pipe([sentence]))
    # print()
    # # print()
    #
    #
    # # print('\n')
    # res3 = pipe(['Tom brings the dog to the park.'])
    # print(res3)
    # print()
    #
    #
    # res3 = pipe(['.', 'Tom brings the dog.' ])
    # print(res3)

