import json
import random
import sys
import heapq
import math
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

def split_data(corpus):
    train, temp = train_test_split(corpus, test_size=0.2, random_state=42)
    eval_set, test_set = train_test_split(temp, test_size=0.5, random_state=42)

    return train, eval_set, test_set

def train_ngram_model(train_data, n):
    train, vocab = padded_everygram_pipeline(n, train_data)
    model = MLE(n)
    model.fit(train, vocab)

    return model

def calculate_perplexity(model, eval_data, n):
    total_perplexity = 0
    count = 0
    
    for sentence in eval_data:
        padded_sentence = ['<s>'] * (n - 1) + sentence + ['</s>']
        n_grams = list(ngrams(padded_sentence, n))
        
        log_prob = 0
        for context, word in [(ngram[:-1], ngram[-1]) for ngram in n_grams]:
            word_prob = model.score(word, context)
            if word_prob > 0:
                log_prob += math.log(word_prob)
            else:
                continue
        
        N = len(sentence) + 1 
        
        if log_prob < 0:  
            sentence_perplexity = math.exp(-log_prob / N)
            total_perplexity += sentence_perplexity
            count += 1
    
    return total_perplexity / count if count > 0 else float('inf')

def predict_next_tokens(model, context, top_k=3):
    probabilities = {word: model.score(word, context) for word in model.vocab if model.score(word, context) > 0}

    return [(word, f"{prob:.3f}") for word, prob in heapq.nlargest(top_k, probabilities.items(), key=lambda x: x[1])]

def complete_methods(model, test_data, n, sample_size=100, random_seed=42):
    random.seed(random_seed)
    sampled_methods = random.sample(test_data, min(sample_size, len(test_data)))
    results = {}
    
    for idx, method in enumerate(tqdm(sampled_methods, desc="Generating JSON Output", unit="method")):
        method_results = []
        for i in range(len(method)):
            context = tuple(['<s>'] * (n - 1) + method[:i])[-(n - 1):]
            predictions = predict_next_tokens(model, context)
            method_results.append(predictions)
        results[str(idx)] = method_results
    
    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: python ngram_script.py <student_corpus.txt> <teacher_corpus.txt>")
        return
    
    corpus_file = sys.argv[1]
    teacher_corpus_file = sys.argv[2]

    corpus = load_corpus(corpus_file)
    teacher_corpus = load_corpus(teacher_corpus_file)
    
    train_data, eval_data, test_data = split_data(corpus)
    
    ngram_values = [3, 5, 7]
    
    models = {n: train_ngram_model(train_data, n) for n in ngram_values}
    teacher_models = {n: train_ngram_model(teacher_corpus, n) for n in ngram_values}
    
    perplexities = {}
    teacher_perplexities = {}
    
    for n in ngram_values:
        perplexities[n] = calculate_perplexity(models[n], eval_data, n)
        teacher_perplexities[n] = calculate_perplexity(teacher_models[n], eval_data, n)
        
        print(f"Perplexity for {n}-gram student model: {perplexities[n]:.3f}")
        print(f"Perplexity for {n}-gram teacher model: {teacher_perplexities[n]:.3f}")
    
    best_n = min(perplexities, key=perplexities.get)
    best_teacher_n = min(teacher_perplexities, key=teacher_perplexities.get)
    
    best_model = models[best_n]
    best_teacher_model = teacher_models[best_teacher_n]
    
    print(f"Best student model: {best_n}-gram with eval perplexity {perplexities[best_n]:.3f}")
    print(f"Best teacher model: {best_teacher_n}-gram with eval perplexity {teacher_perplexities[best_teacher_n]:.3f}")

    perplexities_test = calculate_perplexity(best_model, test_data, best_n)
    perplexities_teacher_test = calculate_perplexity(best_teacher_model, test_data, best_teacher_n)

    print(f"Best student model: {best_n}-gram with test perplexity {perplexities_test:.3f}")
    print(f"Best teacher model: {best_teacher_n}-gram with test perplexity {perplexities_teacher_test:.3f}")
    
    print(f"Testing student {best_n}-gram model")
    main_completion_results = complete_methods(best_model, test_data, best_n)
    print(f"Testing teacher {best_teacher_n}-gram model")
    teacher_completion_results = complete_methods(best_teacher_model, test_data, best_teacher_n)
    
    with open("results_student_model.json", "w", encoding="utf-8") as f:
        json.dump(main_completion_results, f, indent=4)
        
   
    with open("results_teacher_model.json", "w", encoding="utf-8") as f:
        json.dump(teacher_completion_results, f, indent=4)
    
    print("Results saved to results_student_model.json and results_teacher_model.json")

if __name__ == "__main__":
    main()
