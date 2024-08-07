import os
import numpy as np
import gensim.downloader as api
import spacy
from sentence_transformers import SentenceTransformer
from rule_based_extraction import rule_based_extractor

kas = rule_based_extractor()


def extract_lines(file_path, start_marker, end_marker):
    with open(file_path, 'r') as file:
        extract = False
        results = []

        for line in file:
            if start_marker in line:
                extract = True

            if end_marker in line and extract:
                break

            if extract:
                results.append(line.strip()) 

        return results



wv = api.load('word2vec-google-news-300') 
nlp = spacy.load("en_core_web_sm")  

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def framing(sentences):
    '''
        create actions from the sets
    '''
    actions_and_states = []
    for sentence in sentences:
        sentence = sentence.lower()
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == "VERB":
                subjects_or_objects = [child.text for child in token.children if child.dep_ in ["dobj", "nsubj"]]
                if subjects_or_objects:
                    actions_and_states.append([token.text] + subjects_or_objects)
    return actions_and_states



def goal_wise_similarity(truth, pred):
    '''
        Taken in input two sets of actions give the result of the similarity 
    '''
    truth_list = truth.copy()
    pred_list = pred.copy()
    conf = []
    for element in truth_list:
        verb = element[0]
        nouns = element[1:]
        best = None
        max_similarity = 0
        for element2 in pred_list:
            verb2 = element2[0]
            nouns2 = element2[1:]
            verbs_similarity = wv.similarity(verb, verb2)
            nouns_similarity = []
            min_len = min(len(nouns), len(nouns2))
            try:
              if len(nouns) != len(nouns2) and min_len == 0:
                  nouns_similarity = 0
              else:
                  if len(nouns) < len(nouns2):
                      nouns_similarity = [1 if wv.similarity(nouns[i], nouns2[i]) >= 0.708 else 0 for i in range(min_len)]
                  else:
                      nouns_similarity = [1 if wv.similarity(nouns2[i], nouns[i]) >= 0.708 else 0 for i in range(min_len)]
                  if not nouns_similarity:
                      nouns_similarity = 1
                  else:
                      nouns_similarity = np.mean(nouns_similarity)
              if verbs_similarity * nouns_similarity > max_similarity and verbs_similarity > 0.708:
                  max_similarity = verbs_similarity * nouns_similarity
                  best = element2
            except:
                best = None
        if best is not None:
            index_delete = pred_list.index(best)
            pred_list.pop(index_delete)
            conf.append(1)
        else:
            conf.append(0)
    if not conf:
        return 0
    return np.mean(conf)

def sentence_similarity(list1,list2):
    '''
        Taken in input two sets of sentences give the similarity between the plans
    '''
    truth_copy = list1.copy()
    pred_copy = list2.copy()
    conf = []
    for i in range(len(truth_copy)):
        max = float("-inf")
        best = None
        embeddings = model.encode(truth_copy[i])
        for j in range(len(pred_copy)):
            embeddings2 = model.encode(pred_copy[j])
            similarity = np.dot(embeddings,embeddings2)/(np.linalg.norm(embeddings)*np.linalg.norm(embeddings2))
            if similarity > max and similarity > 0.675:
                max = similarity
                best = pred_copy[j]

        if best is not None:
            pred_copy.remove(best)
            conf.append(1)
        else:
            conf.append(0)
    if len(conf) == 0:
        return 0
    return np.mean(conf)


def pg2s(truth,predict):
    truth_actions = framing(truth)
    predict_actions = framing(predict)
    goal_wise = goal_wise_similarity(truth_actions,predict_actions)
    sentence_wise = sentence_similarity(truth,predict)
    pg2s = goal_wise *0.5 + sentence_wise *0.5
    return pg2s


for text_file in os.listdir("./ground_truth"):
      with open("ground_truth/" + text_file,"r") as file:
          lines = file.readlines()
      task = lines[0].replace("Task:","")
      gt_plan = lines[1].replace("Piano:","").replace("STEP","").replace("END","").split("|")
      file_to_open = ("results_save/"+text_file.split("_")[0]+".jpg_"+task)[:-1]
      with open(file_to_open,"a") as f:
            f.write("\n-------------------------------------------------\n")
            f.write("\n\n\nTask: " + task + "\n")
            f.write("Ground Truth: " + str(gt_plan) + "\n")
            
      start_marker = 'SINGLE AGENT 4 TABLE'
      end_marker = '-----' 
      single_table = (extract_lines(file_to_open, start_marker, end_marker)[1:])
      single_table_sim = pg2s(gt_plan,single_table)
      gts = {i: [step] for i, step in enumerate(gt_plan)}
      res = {i: [step] for i, step in enumerate(single_table)}
      try:
        kas_single_tab = (kas.compute_score(gts,res))
      except:
        kas_single_tab = "ERROR"

      start_marker = 'MULTI AGENT 4 TABLE'
      multi_table = (extract_lines(file_to_open, start_marker, end_marker)[1:])
      multi_table_sim = pg2s(gt_plan,multi_table)
      gts = {i: [step] for i, step in enumerate(gt_plan)}
      res = {i: [step] for i, step in enumerate(multi_table)}

      try:
        kas_multi_tab = (kas.compute_score(gts,res))
      except BaseException as e:
        kas_multi_tab = "ERROR"
      
      start_marker = 'SINGLE AGENT 4 VISION'
      single_vision = (extract_lines(file_to_open, start_marker, end_marker)[1:])
      single_vision_sim = pg2s(gt_plan,single_vision)
      gts = {i: [step] for i, step in enumerate(gt_plan)}
      res = {i: [step] for i, step in enumerate(single_vision)}
      try:
        kas_single_vision = (kas.compute_score(gts,res))
      except:
        kas_single_vision = "ERROR"
      

      start_marker = 'PLANNING AGENT RESPONSE'
      multi_vision = (extract_lines(file_to_open, start_marker, end_marker)[1:])
      multi_vision_sim = pg2s(gt_plan,multi_vision)
      gts = {i: [step] for i, step in enumerate(gt_plan)}
      res = {i: [step] for i, step in enumerate(multi_vision)}
      try:
        multi_vision_kas = (kas.compute_score(gts,res))
      except BaseException as e:
        multi_vision_kas = "ERROR"


      with open("results.txt","a") as f:
          print("aggiungo")
          f.write("\n-------------------------------------------------\n")
          f.write("\n\n\nTask: " + task + "\n")
          f.write("Single Table: " + str(single_table_sim) + "\n")
          f.write("KAS Single Table: " + str(kas_single_tab[0]) + "\n")
          f.write("Multi Table: " + str(multi_table_sim) + "\n")
          f.write("KAS Multi Table: " + str(kas_multi_tab[0]) + "\n")    
          f.write("Single Vision: " + str(single_vision_sim) + "\n")
          f.write("KAS Single Vision: " + str(kas_single_vision[0]) + "\n")
          f.write("Multi Vision: " + str(multi_vision_sim) + "\n")
          f.write("KAS Multi Vision: " + str(multi_vision_kas[0]) + "\n")

