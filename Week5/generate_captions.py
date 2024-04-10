"""
from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
gen2 = pipeline('text-generation', model="openai-community/gpt2")

nouns = 'dogs'
verbs = 'playing'

prompt = f'Describe an image where appears {nouns} and {verbs}'
res = generator(prompt, max_length=50, do_sample=True, truncation=True, temperature=0.9, num_workers=4)
res2 = gen2(prompt, max_length=50, truncation=True, do_sample=True, temperature=10, num_workers=4)

print(res)
print(res2)

"""

import random

import pytorch_lightning as pl
import torch

# Set seed globally
from keytotext import pipeline

nlp = pipeline("k2t-base")
pl.seed_everything(34)
keywords = ["man", "skateboard", "dogs", "playing"]

print(nlp(keywords))
