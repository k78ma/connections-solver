import os
import random
import argparse
import itertools
from typing import List
# from tqdm import tqdm

import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

from deap import base, creator, tools, algorithms

parser = argparse.ArgumentParser(description='NYT Connections Solver')
parser.add_argument('filename', type=str, help='Filename of the puzzle to solve')

def load_model():
    print("Loading model...")
    if os.path.exists("word2vec.model"):
        model = KeyedVectors.load("word2vec.model")
    else:
        model = api.load('fasttext-wiki-news-subwords-300')
        model.save("word2vec.model")
    return model

def load_puzzle(filename: str) -> List[str]:
    try:
        with open(f'puzzles/{filename}', 'r') as file:
            word_list = [line.strip() for line in file.readlines()]
        return word_list
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        exit(1)

def calculate_similarity_matrix(word_list: List[str], model: KeyedVectors) -> dict:
    print("Calculating similarities...")
    similarity_matrix = {}
    for i, word1 in enumerate(word_list):
        for j, word2 in enumerate(word_list):
            if i < j:  # To avoid repeating comparisons and comparing a word with itself
                similarity_score = model.similarity(word1, word2)
                similarity_matrix[f"{word1}-{word2}"] = similarity_score
                print(f"Similarity score between '{word1}' and '{word2}': {similarity_score}")
    return similarity_matrix

def group_coherence(group_indices: List[int], words: List[str], matrix: dict) -> float:
    group_words = [words[i] for i in group_indices]
    scores = []
    for word1, word2 in itertools.combinations(group_words, 2):
        key = f"{word1}-{word2}" if f"{word1}-{word2}" in matrix else f"{word2}-{word1}"
        scores.append(matrix[key])
    return np.mean(scores)

def evalGroups(individual, words, similarity_matrix):
    groups = [individual[i*4:(i+1)*4] for i in range(4)]
    return (np.mean([group_coherence(group, words, similarity_matrix) for group in groups]),)

def setup_genetic_algorithm(toolbox, words, similarity_matrix):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox.register("indices", random.sample, range(len(words)), len(words))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalGroups, words=words, similarity_matrix=similarity_matrix)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

def run_genetic_algorithm(toolbox, n=300, cxpb=0.7, mutpb=0.2, ngen=100):
    population = toolbox.population(n)
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)
    return population

def extract_top_individuals(population, word_list, top_n=3):
    top_individuals = tools.selBest(population, top_n)
    for rank, individual in enumerate(top_individuals, start=1):
        result_groups = [sorted([word_list[i] for i in individual[j*4:(j+1)*4]]) for j in range(4)]
        print(f"Best Groups {rank}:", result_groups)

def main():
    args = parser.parse_args()
    model = load_model()
    word_list = load_puzzle(args.filename)
    similarity_matrix = calculate_similarity_matrix(word_list, model)
    
    toolbox = base.Toolbox()
    setup_genetic_algorithm(toolbox, word_list, similarity_matrix)
    population = run_genetic_algorithm(toolbox)
    extract_top_individuals(population, word_list)

if __name__ == "__main__":
    main()
