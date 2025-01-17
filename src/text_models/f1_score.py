"""
This file is used to calculate the f1 score for each file and save the results in the file.
"""
import json

# read the files
with open('../results/bert/caption_bert-base-multilingual-uncased-sentiment_results.json', 'r') as file:
    caption = json.load(file)

with open('../results/bert/description_bert-base-multilingual-uncased-sentiment_results.json', 'r') as file:
    description = json.load(file)

with open('../results/bert/reformulated_caption_bert-base-multilingual-uncased-sentiment_results.json', 'r') as file:
    reformulated_caption = json.load(file)

# get the precision, recall and based on them calculate the f1 score
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

# calculate the f1 score for each file
caption_f1 = f1_score(caption['test_precision'], caption['test_recall'])
description_f1 = f1_score(description['test_precision'], description['test_recall'])
reformulated_caption_f1 = f1_score(reformulated_caption['test_precision'], reformulated_caption['test_recall'])

# save the results in each file
caption['test_f1'] = caption_f1
description['test_f1'] = description_f1
reformulated_caption['test_f1'] = reformulated_caption_f1

# save the files
with open('../results/bert/caption_bert-base-multilingual-uncased-sentiment_results.json', 'w') as file:
    json.dump(caption, file)

with open('../results/bert/description_bert-base-multilingual-uncased-sentiment_results.json', 'w') as file:
    json.dump(description, file)

with open('../results/bert/reformulated_caption_bert-base-multilingual-uncased-sentiment_results.json', 'w') as file:
    json.dump(reformulated_caption, file)