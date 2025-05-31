import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
import itertools
from typing import List, Tuple, Dict, Set, Optional
from tqdm import tqdm
import os
from collections import defaultdict, Counter
import re
from gleam import GLEAM
import esm
import warnings
warnings.filterwarnings('ignore')

REFERENCE_GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


class MLDEOptimizer:
    def __init__(
        self, 
        gleam_model: GLEAM,
        wt_sequence: str,
        max_mutations: int = 6,
        k_mer_size: int = 1
    ):
        self.gleam_model = gleam_model
        self.wt_sequence = wt_sequence
        self.max_mutations = max_mutations
        self.k_mer_size = k_mer_size
        self.device = gleam_model.device
        
        self.esm_model = gleam_model.esm_model
        self.esm_alphabet = gleam_model.esm_alphabet
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        
        self.prediction_cache = {}
        self.all_results = []
        self.corpus_kmers = defaultdict(int)
        self.sequence_corpus = []
        
    def build_corpus_statistics(self, sequences: List[str]):
        self.corpus_kmers.clear()
        self.sequence_corpus = sequences.copy()
        
        for seq in sequences:
            for i in range(len(seq) - self.k_mer_size + 1):
                k_mer = seq[i:i + self.k_mer_size]
                self.corpus_kmers[k_mer] += 1
    
    def calculate_tf_idf_weights(self, sequence: str) -> Dict[str, float]:
        sequence_kmers = defaultdict(int)
        total_kmers_in_seq = 0
        
        for i in range(len(sequence) - self.k_mer_size + 1):
            k_mer = sequence[i:i + self.k_mer_size]
            sequence_kmers[k_mer] += 1
            total_kmers_in_seq += 1
        
        tf_idf_weights = {}
        N = len(self.sequence_corpus)
        
        for k_mer, freq in sequence_kmers.items():
            tf = freq / total_kmers_in_seq
            
            doc_freq = sum(1 for seq in self.sequence_corpus if k_mer in seq)
            idf = np.log(N / (1 + doc_freq))
            
            tf_idf_weights[k_mer] = tf * idf
        
        return tf_idf_weights
    
    def calculate_kmer_entropy(self) -> Dict[str, float]:
        total_kmers = sum(self.corpus_kmers.values())
        entropy_scores = {}
        
        for k_mer, freq in self.corpus_kmers.items():
            p = freq / total_kmers
            entropy_scores[k_mer] = -p * np.log(p + 1e-8)
        
        return entropy_scores
    
    def calculate_kmer_importance(self, sequence: str) -> Dict[int, float]:
        tf_idf_weights = self.calculate_tf_idf_weights(sequence)
        entropy_scores = self.calculate_kmer_entropy()
        
        tf_idf_sum = sum(tf_idf_weights.values()) + 1e-8
        entropy_sum = sum(entropy_scores.values()) + 1e-8
        
        position_importance = {}
        
        for i in range(len(sequence) - self.k_mer_size + 1):
            k_mer = sequence[i:i + self.k_mer_size]
            
            norm_tf_idf = tf_idf_weights.get(k_mer, 0) / tf_idf_sum
            norm_entropy = entropy_scores.get(k_mer, 0) / entropy_sum
            
            importance = norm_tf_idf + norm_entropy
            position_importance[i] = importance
        
        return position_importance
    
    def random_mask(self, sequences: List[str], k_mer_size: int) -> List[Tuple[str, int]]:
        masked_data = []
        
        for seq in sequences:
            if len(seq) < k_mer_size:
                masked_data.append((seq, -1))
                continue
                
            available_positions = list(range(len(seq) - k_mer_size + 1))
            
            if available_positions:
                mask_pos = random.choice(available_positions)
                # Use special marker for masked position
                masked_seq = seq[:mask_pos] + 'X' * k_mer_size + seq[mask_pos + k_mer_size:]
                masked_data.append((masked_seq, mask_pos))
            else:
                masked_data.append((seq, -1))
        
        return masked_data
    
    def importance_mask(self, sequences: List[str], k_mer_size: int) -> List[Tuple[str, int]]:
        masked_data = []
        
        for seq in sequences:
            if len(seq) < k_mer_size:
                masked_data.append((seq, -1))
                continue
                
            importance_scores = self.calculate_kmer_importance(seq)
            
            if importance_scores:
                min_importance_pos = min(importance_scores.keys(), 
                                       key=lambda x: importance_scores[x])
                
                # Use special marker for masked position
                masked_seq = seq[:min_importance_pos] + 'X' * k_mer_size + seq[min_importance_pos + k_mer_size:]
                masked_data.append((masked_seq, min_importance_pos))
            else:
                masked_data.append((seq, -1))
        
        return masked_data
    
    def predict_mutations_esm(self, masked_data: List[Tuple[str, int]]) -> List[str]:
        if not masked_data:
            return []
        
        predicted_sequences = []
        
        for original_seq, mask_pos in masked_data:
            if mask_pos == -1 or 'X' not in original_seq:
                predicted_sequences.append(original_seq.replace('X', random.choice(self.amino_acids)))
                continue
            
            esm_seq = original_seq.replace('X' * self.k_mer_size, '<mask>')
            
            batch_converter = self.esm_alphabet.get_batch_converter()
            batch_data = [("seq", esm_seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                outputs = self.esm_model(batch_tokens, repr_layers=[36])
                logits = outputs['logits']

                mask_token_idx = self.esm_alphabet.mask_idx
                token_positions = (batch_tokens[0] == mask_token_idx).nonzero(as_tuple=True)[0]

                token_pos = token_positions[0].item()
                token_logits = logits[0, token_pos]
                probabilities = F.softmax(token_logits, dim=-1)

                top_idx = torch.argmax(probabilities).item()
                predicted_aa = self.esm_alphabet.get_tok(top_idx)

                predicted_seq = original_seq.replace('X' * self.k_mer_size, predicted_aa * self.k_mer_size)
                predicted_sequences.append(predicted_seq)
        
        return predicted_sequences
    
    def mlde_iteration(
        self, 
        population: List[str], 
        beam_size: int, 
        random_ratio: float,
        iteration: int
    ) -> List[str]:        
        # Step 1: Duplicate sequences for beam search
        expanded_population = []
        for seq in population:
            expanded_population.extend([seq] * beam_size)
        
        # Step 2: Shuffle population
        random.shuffle(expanded_population)
        
        # Step 3: Split into random and importance masking groups
        split_point = int(len(expanded_population) * random_ratio)
        random_group = expanded_population[:split_point]
        importance_group = expanded_population[split_point:]
        
        # Step 4: Apply masking strategies
        masked_data = []
        
        if random_group:
            masked_random = self.random_mask(random_group, self.k_mer_size)
            masked_data.extend(masked_random)
        
        if importance_group:
            masked_importance = self.importance_mask(importance_group, self.k_mer_size)
            masked_data.extend(masked_importance)
        
        # Step 5: Predict mutations using ESM-2
        mutated_sequences = self.predict_mutations_esm(masked_data)
        
        # Step 6: Predict fitness using GLEAM batch prediction
        if mutated_sequences:
            try:
                fitness_results = self.gleam_model.predict_batch(
                    mutated_sequences,
                    batch_size=64
                )
                
                all_candidates = []
                
                for i, result in enumerate(fitness_results):
                    if result.get('predicted_brightness') is not None:
                        seq = result['sequence']
                        fitness = result['predicted_brightness']
                        all_candidates.append((seq, fitness))
                        
                        mutations = self.get_mutations_from_sequences(self.wt_sequence, seq)
                        self.save_results(seq, fitness, mutations, iteration)
                
                for seq in population:
                    if seq not in self.prediction_cache:
                        result = self.gleam_model.predict_single(seq)
                        self.prediction_cache[seq] = result['predicted_brightness']
                    
                    all_candidates.append((seq, self.prediction_cache[seq]))
                
            except Exception as e:
                print(f"    Error in batch prediction: {e}", flush=True)
                all_candidates = [(seq, self.prediction_cache.get(seq, 0.0)) for seq in population]
        else:
            all_candidates = [(seq, self.prediction_cache.get(seq, 0.0)) for seq in population]
        
        # Step 7: Select top-k candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        next_population = [seq for seq, _ in all_candidates[:len(population)]]
        
        return next_population
    
    def get_mutations_from_sequences(self, wt_seq: str, mut_seq: str) -> List[Tuple[str, int, str]]:
        mutations = []
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
            if wt_aa != mut_aa:
                mutations.append((wt_aa, i, mut_aa))
        return mutations
    
    def save_results(self, sequence: str, brightness: float, mutations: List[Tuple[str, int, str]], iteration: int):
        mutation_string = ':'.join([f"{wt}{pos+1}{mut}" for wt, pos, mut in mutations]) if mutations else 'WT'
        
        result = {
            'sequence': sequence,
            'brightness': brightness,
            'mutations': mutations,
            'num_mutations': len(mutations),
            'mutation_string': mutation_string,
            'iteration': iteration,
            'distance_from_wt': sum(1 for i, (wt_aa, mut_aa) in enumerate(zip(self.wt_sequence, sequence)) if wt_aa != mut_aa)
        }
        
        self.all_results.append(result)
    
    def optimize(
        self, 
        population_size: int = 128,
        beam_size: int = 4,
        num_iterations: int = 20,
        random_ratio: float = 0.6,
        early_stopping_patience: int = 10
    ) -> List[Dict]:        
        current_population = [self.wt_sequence] * population_size
        
        self.build_corpus_statistics(current_population)
        
        wt_result = self.gleam_model.predict_single(self.wt_sequence)
        self.prediction_cache[self.wt_sequence] = wt_result['predicted_brightness']
        
        best_fitness = wt_result['predicted_brightness']
        patience_counter = 0
        
        print(f"\nWild-type fitness: {best_fitness:.4f}", flush=True)
        print(f"Population size: {population_size}, Beam size: {beam_size}", flush=True)
        print(f"Random ratio: {random_ratio}, K-mer size: {self.k_mer_size}", flush=True)
        
        for iteration in range(num_iterations):
            print(f"\nIteration: {iteration}", flush=True)
            
            self.build_corpus_statistics(current_population)
            
            # Run MLDE iteration
            current_population = self.mlde_iteration(
                current_population, 
                beam_size, 
                random_ratio,
                iteration
            )
            
            population_fitness = []
            for seq in current_population:
                if seq not in self.prediction_cache:
                    result = self.gleam_model.predict_single(seq)
                    self.prediction_cache[seq] = result['predicted_brightness']
                population_fitness.append(self.prediction_cache[seq])
            
            current_best = max(population_fitness)
            avg_fitness = np.mean(population_fitness)
            unique_seqs = len(set(current_population))
            
            print(f"Best fitness: {current_best:.4f}, mean fitness: {avg_fitness:.4f}, unique: {unique_seqs}, patience: {patience_counter}", flush=True)

            if current_best > best_fitness:
                improvement = current_best - best_fitness
                best_fitness = current_best
                patience_counter = 0
                print(f"    New best: {current_best:.4f} (improvement: +{improvement:.4f})", flush=True)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"    Early stopping at iteration {iteration + 1}", flush=True)
                break
        
        best_variants = []
        final_fitness = []
        
        for seq in set(current_population):
            fitness = self.prediction_cache[seq]
            mutations = self.get_mutations_from_sequences(self.wt_sequence, seq)
            
            if len(mutations) <= self.max_mutations:
                mutation_string = ':'.join([f"{wt}{pos+1}{mut}" for wt, pos, mut in mutations]) if mutations else 'WT'
                variant_info = {
                    'sequence': seq,
                    'brightness': fitness,
                    'mutations': mutations,
                    'num_mutations': len(mutations),
                    'mutation_string': mutation_string
                }
                best_variants.append(variant_info)
                final_fitness.append(fitness)
        
        best_variants.sort(key=lambda x: x['brightness'], reverse=True)
        
        print(f"\nOptimization summary:", flush=True)
        print(f"  Total candidates evaluated: {len(self.all_results)}", flush=True)
        print(f"  Final variants: {len(best_variants)}", flush=True)
        print(f"  Cache size: {len(self.prediction_cache)}", flush=True)
        print(f"  Best fitness achieved: {max(final_fitness) if final_fitness else best_fitness:.4f}", flush=True)
        
        return best_variants
    
    def get_all_results_df(self) -> pd.DataFrame:
        results_data = []
        for result in self.all_results:
            results_data.append({
                'sequence': result['sequence'],
                'brightness': result['brightness'],
                'mutation_string': result['mutation_string'],
                'num_mutations': result['num_mutations'],
                'iteration': result['iteration'],
                'distance_from_wt': result['distance_from_wt']
            })
        
        return pd.DataFrame(results_data)


if __name__ == "__main__":    
    output_path = ... # Path to mlde_optimization_results.csv
    all_results_path = ... # Path to 'mlde_all_results.csv'
    
    gleam_model = GLEAM(
        model_path=..., # Path to gleam.pt
        device=... # 'cuda' or 'cpu' or None for auto detection
    ) 

    optimizer = MLDEOptimizer(
        gleam_model=gleam_model,
        wt_sequence=REFERENCE_GFP,
        max_mutations=6,
        k_mer_size=1
    )
    
    best_variants = optimizer.optimize(
        population_size=128,
        beam_size=4,
        num_iterations=100,
        random_ratio=0.6,
        early_stopping_patience=10
    )
    
    results_data = []
    for variant in best_variants:
        results_data.append({
            'mutations': variant['mutation_string'],
            'predicted_brightness': variant['brightness'],
            'num_mutations': variant['num_mutations'],
            'full_sequence': variant['sequence']
        })
        
    results_df = pd.DataFrame(results_data)
    all_results_df = optimizer.get_all_results_df()

    results_df.to_csv(output_path, index=False)
    all_results_df.to_csv(all_results_path, index=False)
    
    print(f"\nOptimization complete!", flush=True)