from end_to_end_pipeline import QASemEndToEndPipeline
import spacy


if __name__ == "__main__":
    pipe = QASemEndToEndPipeline(nominalization_detection_threshold=0.75)
    sentences = ["The doctor was interested in Luke 's treatment .", "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."] #, "The Veterinary student was interested in Luke 's treatment of sea animals .", "Tom brings the dog to the park."]
    outputs = pipe(sentences)
    #
    print(outputs)

    # pipe = QASemEndToEndPipeline(detection_threshold=0.75)
    sentence = "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."
    print(pipe([sentence]))
    # print()
    res1 = pipe(["The student was interested in Luke 's research about see animals ."])
    print(res1)
    print()#, verb_form="research", predicate_type="nominal")
    res2 = pipe(["The doctor was interested in Luke 's treatment .", "The Veterinary student was interested in Luke 's treatment of sea animals ."])#, verb_form="treat", predicate_type="nominal", num_beams=10)
    print(res2)
    # # #res3 = pipe(["A number of professions have developed that specialize in the treatment of mental disorders ."])
    # # # print(res1)
    #print(res2)
    print('\n')
    res3 = pipe(['Tom brings the dog to the park.'])
    print(res3)