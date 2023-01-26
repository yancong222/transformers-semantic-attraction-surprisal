!pip install minicons
import pandas as pd
import numpy as np
import re
import math
import os
import string
import shutil, sys, glob
from minicons import scorer

la = pd.read_csv('laurinavichyute_dataset_minicons.csv', index_col = 0)
cn = pd.read_csv('cunnings_dataset_dataset_minicons.csv', index_col = 0)

ilm_model = scorer.IncrementalLMScorer('gpt2', 'cpu') 
distilgpt2 = scorer.IncrementalLMScorer('distilgpt2', 'cpu')
gpt_neo = scorer.IncrementalLMScorer('EleutherAI/gpt-neo-1.3B', 'cpu')

def ilm_sent_mean_surp(stimuli):
  lst = []
  lst.append(stimuli)
  score = ilm_model.sequence_score(lst, reduction = lambda x: -x.mean(0).item())
  return round(score[0], 2)

def gptneo_sent_mean_surp(stimuli):
  lst = []
  lst.append(stimuli)
  score = gpt_neo.sequence_score(lst, reduction = lambda x: -x.mean(0).item())
  return round(score[0], 2)

def distilgpt2_sent_mean_surp(stimuli):
  lst = []
  lst.append(stimuli)
  score = distilgpt2.sequence_score(lst, reduction = lambda x: -x.mean(0).item())
  return round(score[0], 2)

la['ilm_sent_mean_surp'] = la['Sentence'].apply(lambda x: ilm_sent_mean_surp(x))
la['distilgpt2_sent_mean_surp'] = la['Sentence'].apply(lambda x: distilgpt2_sent_mean_surp(x))
la['gptneo_sent_mean_surp'] = la['Sentence'].apply(lambda x: gptneo_sent_mean_surp(x))
cn['ilm_sent_mean_surp'] = cn['Sentence'].apply(lambda x: ilm_sent_mean_surp(x))
cn['distilgpt2_sent_mean_surp'] = cn['Sentence'].apply(lambda x: distilgpt2_sent_mean_surp(x))
cn['gptneo_sent_mean_surp'] = cn['Sentence'].apply(lambda x: gptneo_sent_mean_surp(x))

def target_surp(model, sentence, target):
  temp = []
  temp.append(sentence)
  surp = model.token_score(temp, surprisal = True, base_two = True)
  tuple_list = surp[0]
  result = [y[1] for x, y in enumerate(tuple_list) if y[0] == target]
  if len(result) == 0:
    result = [y[1] for x, y in enumerate(tuple_list) if y[0] in target]
  return round(np.nanmean(result), 2)

la['ilm_target_mean_surp'] = la.apply(lambda x: target_surp(ilm_model, x.Sentence, x.Target), axis=1)
la['distilgpt2_target_mean_surp'] = la.apply(lambda x: target_surp(distilgpt2, x.Sentence, x.Target), axis=1)
la['gptneo_target_mean_surp'] = la.apply(lambda x: target_surp(gpt_neo, x.Sentence, x.Target), axis=1)
cn['ilm_target_mean_surp'] = cn.apply(lambda x: target_surp(ilm_model, x.Sentence, x.Target), axis = 1)
cn['distilgpt2_target_mean_surp'] = cn.apply(lambda x: target_surp(distilgpt2, x.Sentence, x.Target), axis=1)
cn['gptneo_target_mean_surp'] = cn.apply(lambda x: target_surp(gpt_neo, x.Sentence, x.Target), axis=1)

def get_tokens_len_score(sent, model):
  input = []
  input.append(sent)
  output = model.token_score(input, surprisal = True, base_two = True)
  result = []
  result.append([len(output[0]), output[0]])
  return result

models = [('ilm', gpt2), ('distilgpt2', distilgpt2), ('gptneo', gpt_neo)]
for model in models:
  la[model[0] + '_len_tokens'] = la['Sentence'].apply(lambda x: get_tokens_len_score(x, model[1])[0][0])
  la[model[0] + '_tokens_surpscore'] = la['Sentence'].apply(lambda x: get_tokens_len_score(x, model[1])[0][1])
  
models = [('ilm', gpt2), ('distilgpt2', distilgpt2), ('gptneo', gpt_neo)]
for model in models:
  cn[model[0] + '_len_tokens'] = cn['Sentence'].apply(lambda x: get_tokens_len_score(x, model[1])[0][0])
  cn[model[0] + '_tokens_surpscore'] = cn['Sentence'].apply(lambda x: get_tokens_len_score(x, model[1])[0][1])





