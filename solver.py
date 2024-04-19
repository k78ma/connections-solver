import os
import random
from collections import defaultdict
from tqdm import tqdm
from deap import base, creator, tools, algorithms
import numpy as np
import itertools

import gensim.downloader as api
from gensim.models import KeyedVectors


print("Loading model...")
if os.path.exists("word2vec.model"):
    model = KeyedVectors.load("word2vec.model")
else:
    model = api.load('fasttext-wiki-news-subwords-300')
    model.save("word2vec.model")

words = ['cal', 'gal', 'in', 'oz', 'aim', 'intend', 'mean', 'plan', 'girls', 'rule', 'grate', 'fleece', 'gutter', 'parachute', 'curb', 'manhole']

print("Calculating similarities...")
similarity_matrix = {}
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:  # To avoid repeating comparisons and comparing a word with itself
            similarity_score = model.similarity(word1, word2)
            similarity_matrix[f"{word1}-{word2}"] = similarity_score
            print(f"Similarity score between '{word1}' and '{word2}': {similarity_score}")


# Helper function to calculate the coherence of a group
def group_coherence(group_indices, words, matrix):
    group_words = [words[i] for i in group_indices]
    scores = []
    for word1, word2 in itertools.combinations(group_words, 2):
        key = f"{word1}-{word2}" if f"{word1}-{word2}" in matrix else f"{word2}-{word1}"
        scores.append(matrix[key])
    return np.mean(scores)

previous_solutions = defaultdict(int)

def diversity_penalty(individual):
    """Calculate how often similar groupings have appeared and penalize commonly occurring patterns."""
    key = tuple(sorted(individual))
    count = previous_solutions[key]
    previous_solutions[key] += 1
    return count

# Fitness function to maximize
def evalGroups(individual):
    # Split individual into four groups of indices
    groups = [individual[i*4:(i+1)*4] for i in range(4)]
    # Calculate and return the average coherence of groups
    return (np.mean([group_coherence(group, words, similarity_matrix) for group in groups]),)

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(words)), len(words))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalGroups)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Running the genetic algorithm
population = toolbox.population(n=300)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=True)

# Extract top 3 results
top_individuals = tools.selBest(population, 3)
for rank, individual in enumerate(top_individuals, start=1):
    result_groups = [sorted([words[i] for i in individual[j*4:(j+1)*4]]) for j in range(4)]
    print(f"Best Groups {rank}:", result_groups)
