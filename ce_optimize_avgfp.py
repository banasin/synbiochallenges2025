import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
from scipy.special import softmax
from tqdm import tqdm
import itertools
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import json
from datetime import datetime
from gleam import GLEAM


REFERENCE_GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

@dataclass
class Mutation:
    position: int
    original_aa: str
    new_aa: str
    
    def __hash__(self):
        return hash((self.position, self.original_aa, self.new_aa))
    
    def __eq__(self, other):
        return (self.position == other.position and 
                self.original_aa == other.original_aa and 
                self.new_aa == other.new_aa)
    
    def __repr__(self):
        return f"{self.original_aa}{self.position+1}{self.new_aa}"

@dataclass
class Candidate:
    mutations: Set[Mutation]
    sequence: str
    fitness: float = 0.0
    
    def get_mutation_positions(self) -> Set[int]:
        return {m.position for m in self.mutations}
    
    def to_mutation_vector(self, target_positions: List[int]) -> np.ndarray:
        vec = np.zeros((len(target_positions), 20))
        pos_to_idx = {pos: i for i, pos in enumerate(target_positions)}
        
        for mut in self.mutations:
            if mut.position in pos_to_idx:
                aa_idx = AMINO_ACIDS.index(mut.new_aa)
                vec[pos_to_idx[mut.position], aa_idx] = 1
                
        return vec

class EpistaticCrossEntropy:
    def __init__(self, 
                 gleam_predictor,
                 target_positions: List[int],
                 reference_sequence: str = REFERENCE_GFP,
                 population_size: int = 10000,
                 elite_fraction: float = 0.2,
                 smoothing_alpha: float = 0.7,
                 max_mutations: int = 6,
                 temperature: float = 1.0,
                 epistasis_threshold: float = 0.1,
                 device: Optional[str] = None):
        
        self.gleam = gleam_predictor
        self.target_positions = sorted(target_positions)
        self.reference_seq = reference_sequence
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.smoothing_alpha = smoothing_alpha
        self.max_mutations = max_mutations
        self.temperature = temperature
        self.epistasis_threshold = epistasis_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.n_positions = len(target_positions)
        self.n_aa = 20
        
        self.single_probs = np.ones((self.n_positions, self.n_aa)) / self.n_aa
        
        self.pair_interactions = np.zeros((self.n_positions, self.n_aa, 
                                          self.n_positions, self.n_aa))
        
        # Optional, memory intensive
        self.use_triple_interactions = False
        if self.use_triple_interactions:
            self.triple_interactions = np.zeros((self.n_positions, self.n_aa,
                                               self.n_positions, self.n_aa,
                                               self.n_positions, self.n_aa))
        
        self.n_mutations_probs = np.array([0.0] + [1.0/max_mutations] * max_mutations)
        self.n_mutations_probs /= self.n_mutations_probs.sum()
        
        self.elite_memory = []
        self.generation = 0
        
        self.pos_to_idx = {pos: i for i, pos in enumerate(target_positions)}
        
        self._mask_wildtype()
        
    def _mask_wildtype(self):
        for i, pos in enumerate(self.target_positions):
            wt_aa = self.reference_seq[pos]
            if wt_aa in AMINO_ACIDS:
                aa_idx = AMINO_ACIDS.index(wt_aa)
                self.single_probs[i, aa_idx] = 0.0
                
        for i in range(self.n_positions):
            if self.single_probs[i].sum() > 0:
                self.single_probs[i] /= self.single_probs[i].sum()
    
    def apply_mutations(self, mutations: Set[Mutation]) -> str:
        seq_list = list(self.reference_seq)
        for mut in mutations:
            seq_list[mut.position] = mut.new_aa
        return ''.join(seq_list)
    
    def sample_candidate_advanced(self) -> Candidate:
        exploration_rate = max(0.1, 0.3 * np.exp(-self.generation / 10))
        
        if np.random.rand() < exploration_rate:
            if np.random.rand() < 0.5:
                n_mut = np.random.choice(range(3, min(6, self.max_mutations) + 1))
            else:
                n_mut = np.random.choice(range(1, self.max_mutations + 1))
        else:
            n_mut = np.random.choice(range(1, self.max_mutations + 1), 
                                   p=self.n_mutations_probs[1:] / self.n_mutations_probs[1:].sum())
        
        mutations = set()
        selected_positions = set()
        
        use_uniform = np.random.rand() < exploration_rate * 0.5
        
        for i in range(n_mut):
            if use_uniform:
                available = [p for p in self.target_positions if p not in selected_positions]
                if not available:
                    break
                position = np.random.choice(available)
                pos_idx = self.pos_to_idx[position]
            else:
                pos_probs = self._calculate_position_probs(mutations, selected_positions)
                
                if pos_probs.sum() == 0:
                    break
                    
                pos_temp = 0.5 + 0.5 * np.exp(-self.generation / 20)
                pos_probs = np.power(pos_probs, 1.0 / pos_temp)
                pos_probs /= pos_probs.sum()
                
                pos_idx = np.random.choice(self.n_positions, p=pos_probs)
                position = self.target_positions[pos_idx]
            
            selected_positions.add(position)
            
            aa_probs = self._calculate_aa_probs(pos_idx, mutations)
            
            if np.random.rand() < exploration_rate * 0.3:
                valid_aas = aa_probs > 0
                uniform = np.zeros_like(aa_probs)
                uniform[valid_aas] = 1.0 / np.sum(valid_aas)
                aa_probs = 0.7 * aa_probs + 0.3 * uniform
                aa_probs /= aa_probs.sum()
            
            aa_idx = np.random.choice(self.n_aa, p=aa_probs)
            new_aa = AMINO_ACIDS[aa_idx]
            
            mutation = Mutation(position, self.reference_seq[position], new_aa)
            mutations.add(mutation)
        
        sequence = self.apply_mutations(mutations)
        return Candidate(mutations, sequence)
    
    def _calculate_position_probs(self, existing_mutations: Set[Mutation], 
                                 selected_positions: Set[int]) -> np.ndarray:
        probs = np.ones(self.n_positions)
        
        for pos in selected_positions:
            if pos in self.pos_to_idx:
                probs[self.pos_to_idx[pos]] = 0.0
        
        if len(existing_mutations) > 0 and np.any(self.pair_interactions != 0):
            for mut in existing_mutations:
                if mut.position in self.pos_to_idx:
                    mut_pos_idx = self.pos_to_idx[mut.position]
                    mut_aa_idx = AMINO_ACIDS.index(mut.new_aa)
                    
                    for pos_idx in range(self.n_positions):
                        if probs[pos_idx] > 0:
                            interaction_score = np.mean(
                                self.pair_interactions[mut_pos_idx, mut_aa_idx, pos_idx, :]
                            )
                            probs[pos_idx] *= np.exp(interaction_score / self.temperature)
        
        if probs.sum() > 0:
            probs /= probs.sum()
        
        return probs
    
    def _calculate_aa_probs(self, pos_idx: int, existing_mutations: Set[Mutation]) -> np.ndarray:
        probs = self.single_probs[pos_idx].copy()
        
        if len(existing_mutations) > 0 and np.any(self.pair_interactions != 0):
            for mut in existing_mutations:
                if mut.position in self.pos_to_idx:
                    mut_pos_idx = self.pos_to_idx[mut.position]
                    mut_aa_idx = AMINO_ACIDS.index(mut.new_aa)
                    
                    interaction_scores = self.pair_interactions[mut_pos_idx, mut_aa_idx, pos_idx, :]
                    probs *= np.exp(interaction_scores / self.temperature)
        
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(self.n_aa) / self.n_aa
            
        return probs
    
    def evaluate_batch(self, candidates: List[Candidate], batch_size: int = 32) -> List[Candidate]:
        sequences = [c.sequence for c in candidates]
        
        results = self.gleam.predict_batch(
            sequences, 
            return_uncertainty=True,
            batch_size=batch_size
        )
        
        for candidate, result in zip(candidates, results):
            candidate.fitness = result['predicted_brightness']
            
        return candidates
    
    def update_distributions(self, elite_candidates: List[Candidate]):
        new_single_probs = np.zeros_like(self.single_probs)
        
        for candidate in elite_candidates:
            for mut in candidate.mutations:
                if mut.position in self.pos_to_idx:
                    pos_idx = self.pos_to_idx[mut.position]
                    aa_idx = AMINO_ACIDS.index(mut.new_aa)
                    new_single_probs[pos_idx, aa_idx] += 1
        
        temperature_norm = 0.5 + 0.5 * (self.generation / 50)
        
        for i in range(self.n_positions):
            if new_single_probs[i].sum() > 0:
                counts = new_single_probs[i] + 1.0
                probs = np.exp(np.log(counts) / temperature_norm)
                new_single_probs[i] = probs / probs.sum()
            else:
                new_single_probs[i] = self.single_probs[i]
        
        min_entropy = 0.5
        for i in range(self.n_positions):
            p = new_single_probs[i]
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(np.sum(p > 0))
            
            if entropy < min_entropy and max_entropy > min_entropy:
                uniform = np.ones_like(p) / np.sum(p > 0)
                uniform[p == 0] = 0
                alpha_mix = entropy / min_entropy
                new_single_probs[i] = alpha_mix * p + (1 - alpha_mix) * uniform
                new_single_probs[i] /= new_single_probs[i].sum()
        
        adaptive_alpha = min(0.7, 0.3 + 0.02 * self.generation)
        
        self.single_probs = (adaptive_alpha * new_single_probs + 
                            (1 - adaptive_alpha) * self.single_probs)
        
        self._update_pair_interactions(elite_candidates)
        
        mutation_counts = np.zeros(self.max_mutations + 1)
        for candidate in elite_candidates:
            n_mut = len(candidate.mutations)
            mutation_counts[n_mut] += 1
        
        if mutation_counts[1:].sum() > 0:
            mutation_counts += 5.0
            new_n_mut_probs = mutation_counts / mutation_counts.sum()
            
            min_prob = 0.05
            new_n_mut_probs[1:] = np.maximum(new_n_mut_probs[1:], min_prob)
            new_n_mut_probs /= new_n_mut_probs.sum()
            
            self.n_mutations_probs = (0.3 * new_n_mut_probs + 
                                     0.7 * self.n_mutations_probs)
    
    def _update_pair_interactions(self, elite_candidates: List[Candidate]):
        cooccurrence = np.zeros((self.n_positions, self.n_aa, self.n_positions, self.n_aa))
        single_counts = np.zeros((self.n_positions, self.n_aa))
        
        for candidate in elite_candidates:
            mutations = list(candidate.mutations)
            
            for mut in mutations:
                if mut.position in self.pos_to_idx:
                    pos_idx = self.pos_to_idx[mut.position]
                    aa_idx = AMINO_ACIDS.index(mut.new_aa)
                    single_counts[pos_idx, aa_idx] += 1
            
            for i, mut1 in enumerate(mutations):
                for j, mut2 in enumerate(mutations):
                    if i < j and mut1.position in self.pos_to_idx and mut2.position in self.pos_to_idx:
                        pos1_idx = self.pos_to_idx[mut1.position]
                        pos2_idx = self.pos_to_idx[mut2.position]
                        aa1_idx = AMINO_ACIDS.index(mut1.new_aa)
                        aa2_idx = AMINO_ACIDS.index(mut2.new_aa)
                        
                        cooccurrence[pos1_idx, aa1_idx, pos2_idx, aa2_idx] += 1
                        cooccurrence[pos2_idx, aa2_idx, pos1_idx, aa1_idx] += 1
        
        n_elite = len(elite_candidates)
        
        for pos1 in range(self.n_positions):
            for aa1 in range(self.n_aa):
                for pos2 in range(self.n_positions):
                    for aa2 in range(self.n_aa):
                        if pos1 != pos2:
                            expected = (single_counts[pos1, aa1] * single_counts[pos2, aa2]) / (n_elite + 1e-8)
                            
                            observed = cooccurrence[pos1, aa1, pos2, aa2]
                            
                            if expected > 0 and observed > 0:
                                score = np.log((observed + 1) / (expected + 1))
                                
                                self.pair_interactions[pos1, aa1, pos2, aa2] = (
                                    0.8 * self.pair_interactions[pos1, aa1, pos2, aa2] + 
                                    0.2 * score
                                )
    
    def local_search(self, candidates, n_steps: int = 5, subsample_size: int = 100, batch_size: int = 32) -> List[Candidate]:
        single_candidate_mode = False
        if isinstance(candidates, Candidate):
            candidates = [candidates]
            single_candidate_mode = True
        
        best_candidates = candidates.copy()
        
        for step in range(n_steps):
            all_neighbors = []
            neighbor_to_candidate_idx = []
            
            for idx, candidate in enumerate(candidates):
                candidate_neighbors = []
                
                if len(candidate.mutations) > 1:
                    for mut in candidate.mutations:
                        new_mutations = candidate.mutations - {mut}
                        new_seq = self.apply_mutations(new_mutations)
                        candidate_neighbors.append(Candidate(new_mutations, new_seq))
                
                if len(candidate.mutations) < self.max_mutations:
                    used_positions = candidate.get_mutation_positions()
                    for pos in self.target_positions:
                        if pos not in used_positions:
                            for aa in AMINO_ACIDS:
                                if aa != self.reference_seq[pos]:
                                    new_mut = Mutation(pos, self.reference_seq[pos], aa)
                                    new_mutations = candidate.mutations | {new_mut}
                                    new_seq = self.apply_mutations(new_mutations)
                                    candidate_neighbors.append(Candidate(new_mutations, new_seq))
                
                for mut in candidate.mutations:
                    for aa in AMINO_ACIDS:
                        if aa != mut.new_aa and aa != mut.original_aa:
                            new_mutations = (candidate.mutations - {mut}) | {
                                Mutation(mut.position, mut.original_aa, aa)
                            }
                            new_seq = self.apply_mutations(new_mutations)
                            candidate_neighbors.append(Candidate(new_mutations, new_seq))
                
                if subsample_size > 0:
                    candidate_neighbors = np.random.choice(candidate_neighbors, subsample_size, replace=False).tolist()
                                
                all_neighbors.extend(candidate_neighbors)
                neighbor_to_candidate_idx.extend([idx] * len(candidate_neighbors))
            
            if all_neighbors:
                all_neighbors = self.evaluate_batch(all_neighbors, batch_size)
                
                for neighbor, candidate_idx in zip(all_neighbors, neighbor_to_candidate_idx):
                    if neighbor.fitness > best_candidates[candidate_idx].fitness:
                        best_candidates[candidate_idx] = neighbor
                        candidates[candidate_idx] = neighbor
        
        if single_candidate_mode:
            return best_candidates[0]
        else:
            return best_candidates

    def optimize(self, 
                n_generations: int = 50,
                batch_size: int = 32,
                local_search_elite: int = 10,
                subsample_size_elite: int = 100,
                subsample_size_final: int = 100,
                adaptive_temperature: bool = True,
                save_checkpoint: bool = True,
                checkpoint_path: str = "ce_checkpoint.pkl",
                verbose: bool = True) -> Tuple[Candidate, List[Dict]]:        
        history = []
        
        for gen in range(n_generations):
            self.generation = gen
            
            if adaptive_temperature:
                self.temperature = 2.0 * (0.5 ** (gen / n_generations))
            
            if verbose:
                print(f"\nGeneration {gen + 1}/{n_generations}", flush=True)
                print(f"Temperature: {self.temperature:.4f}", flush=True)
            
            candidates = []
            seen_mutations = set()
            duplicates = 0
            
            quota_sizes = {
                1: int(0.2 * self.population_size),
                2: int(0.25 * self.population_size),
                3: int(0.25 * self.population_size),
                4: int(0.15 * self.population_size),
                5: int(0.1 * self.population_size),
                6: int(0.05 * self.population_size)
            }
            
            total_quota = sum(quota_sizes.values())
            if total_quota != self.population_size:
                quota_sizes[2] += self.population_size - total_quota
            
            for n_mut, quota in quota_sizes.items():
                if n_mut > self.max_mutations:
                    continue
                    
                for _ in range(quota):
                    attempts = 0
                    while attempts < 10:
                        candidate = self.sample_candidate_advanced()
                        while len(candidate.mutations) != n_mut and attempts < 5:
                            candidate = self.sample_candidate_advanced()
                            attempts += 1
                        
                        mut_tuple = tuple(sorted(candidate.mutations, key=lambda m: (m.position, m.new_aa)))
                        if mut_tuple not in seen_mutations:
                            seen_mutations.add(mut_tuple)
                            candidates.append(candidate)
                            break
                        else:
                            duplicates += 1
                            attempts += 1
                    
                    if attempts >= 10:
                        if candidates:
                            base_candidate = np.random.choice(candidates[-10:])
                            perturbed = self._perturb_candidate(base_candidate)
                            candidates.append(perturbed)
            
            if verbose and duplicates > 0:
                print(f"Avoided {duplicates} duplicate candidates", flush=True)
            
            while len(candidates) < self.population_size:
                candidates.append(self.sample_candidate_advanced())
            
            if self.elite_memory:
                n_memory = min(len(self.elite_memory), self.population_size // 5)
                
                if len(self.elite_memory) > n_memory:
                    n_top = n_memory // 2
                    n_rest = n_memory - n_top
                    elite_sample = (
                        self.elite_memory[:n_top] + 
                        list(np.random.choice(self.elite_memory[n_top:], n_rest, replace=False))
                    )
                else:
                    elite_sample = self.elite_memory
                
                for elite in elite_sample[:n_memory//2]:
                    candidates.append(elite)
                
                for elite in elite_sample[n_memory//2:]:
                    perturbed = self._perturb_candidate(elite)
                    candidates.append(perturbed)
            
            candidates = candidates[:self.population_size]
            
            if verbose:
                print("Evaluating population...", flush=True)
            candidates = self.evaluate_batch(candidates, batch_size)
            
            candidates.sort(key=lambda x: x.fitness, reverse=True)
            
            dynamic_elite_size = int(self.elite_size * (1.5 - 0.5 * gen / n_generations))
            elite = candidates[:dynamic_elite_size]
            
            if local_search_elite > 0 and gen % 5 == 0:
                if verbose:
                    print(f"Performing local search on top {local_search_elite} candidates...", flush=True)
                elite_to_search = elite[:min(local_search_elite, len(elite))]
                improved_elite = self.local_search(elite_to_search, n_steps=1, subsample_size=subsample_size_elite, batch_size=batch_size)
                
                for i, improved in enumerate(improved_elite):
                    elite[i] = improved
            
            self.update_distributions(elite)
            
            self.elite_memory.extend(elite[:20])
            
            diverse_candidates = self._select_diverse_candidates(candidates[dynamic_elite_size:], n=10)
            self.elite_memory.extend(diverse_candidates)
            
            unique_elite = {}
            for cand in self.elite_memory:
                mut_tuple = tuple(sorted(cand.mutations, key=lambda m: (m.position, m.new_aa)))
                if mut_tuple not in unique_elite or cand.fitness > unique_elite[mut_tuple].fitness:
                    unique_elite[mut_tuple] = cand
            
            self.elite_memory = sorted(unique_elite.values(), key=lambda x: x.fitness, reverse=True)[:100]
            
            fitness_values = [c.fitness for c in candidates]
            elite_fitness = [c.fitness for c in elite]
            
            unique_mutations = len(set(str(m) for c in elite for m in c.mutations))
            avg_n_mutations = np.mean([len(c.mutations) for c in elite])
            
            stats = {
                'generation': gen + 1,
                'mean_fitness': np.mean(fitness_values),
                'max_fitness': np.max(fitness_values),
                'min_fitness': np.min(fitness_values),
                'std_fitness': np.std(fitness_values),
                'elite_mean': np.mean(elite_fitness),
                'elite_std': np.std(elite_fitness),
                'best_candidate': elite[0],
                'temperature': self.temperature,
                'unique_mutations': unique_mutations,
                'avg_n_mutations': avg_n_mutations,
                'duplicates_avoided': duplicates
            }
            
            history.append(stats)
            
            if verbose:
                print(f"Best fitness: {stats['max_fitness']:.4f}", flush=True)
                print(f"Elite mean: {stats['elite_mean']:.4f} ± {stats['elite_std']:.4f}", flush=True)
                print(f"Best mutations: {elite[0].mutations}", flush=True)
                print(f"Unique mutations in elite: {unique_mutations}", flush=True)
                print(f"Avg mutations per candidate: {avg_n_mutations:.2f}", flush=True)
            
            if save_checkpoint and (gen + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_path, history)
        
        best_candidate = self.elite_memory[0]
        if verbose:
            print("\nPerforming final intensive local search...", flush=True)
        best_candidate = self.local_search(best_candidate, n_steps=3, subsample_size=subsample_size_final, batch_size=batch_size)
        
        return best_candidate, history
    
    def _perturb_candidate(self, candidate: Candidate) -> Candidate:
        mutations = candidate.mutations.copy()
        
        perturb_type = np.random.choice(['add', 'remove', 'change', 'swap'])
        
        if perturb_type == 'add' and len(mutations) < self.max_mutations:
            used_positions = candidate.get_mutation_positions()
            available = [p for p in self.target_positions if p not in used_positions]
            if available:
                pos = np.random.choice(available)
                pos_idx = self.pos_to_idx[pos]
                aa_probs = self.single_probs[pos_idx].copy()
                if aa_probs.sum() > 0:
                    aa_idx = np.random.choice(self.n_aa, p=aa_probs)
                    new_aa = AMINO_ACIDS[aa_idx]
                    mutations.add(Mutation(pos, self.reference_seq[pos], new_aa))
        
        elif perturb_type == 'remove' and len(mutations) > 1:
            mut_to_remove = np.random.choice(list(mutations))
            mutations.remove(mut_to_remove)
        
        elif perturb_type == 'change' and mutations:
            mut_to_change = np.random.choice(list(mutations))
            mutations.remove(mut_to_change)
            
            pos_idx = self.pos_to_idx[mut_to_change.position]
            aa_probs = self.single_probs[pos_idx].copy()
            current_aa_idx = AMINO_ACIDS.index(mut_to_change.new_aa)
            aa_probs[current_aa_idx] = 0
            
            if aa_probs.sum() > 0:
                aa_probs /= aa_probs.sum()
                aa_idx = np.random.choice(self.n_aa, p=aa_probs)
                new_aa = AMINO_ACIDS[aa_idx]
                mutations.add(Mutation(mut_to_change.position, mut_to_change.original_aa, new_aa))
            else:
                mutations.add(mut_to_change)
        
        elif perturb_type == 'swap' and len(mutations) >= 2:
            muts_list = list(mutations)
            mut1, mut2 = np.random.choice(muts_list, 2, replace=False)
            mutations.remove(mut1)
            mutations.remove(mut2)
            mutations.add(Mutation(mut1.position, mut1.original_aa, mut2.new_aa))
            mutations.add(Mutation(mut2.position, mut2.original_aa, mut1.new_aa))
        
        sequence = self.apply_mutations(mutations)
        perturbed = Candidate(mutations, sequence)
        perturbed.fitness = candidate.fitness * 0.95
        
        return perturbed
    
    def _select_diverse_candidates(self, candidates: List[Candidate], n: int) -> List[Candidate]:
        if len(candidates) <= n:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        first = np.random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        while len(selected) < n and remaining:
            max_min_dist = -1
            best_candidate = None
            
            for cand in remaining:
                min_dist = float('inf')
                for sel in selected:
                    union = len(cand.mutations | sel.mutations)
                    intersection = len(cand.mutations & sel.mutations)
                    if union > 0:
                        dist = 1 - intersection / union
                    else:
                        dist = 1
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = cand
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def save_checkpoint(self, path: str, history: List[Dict]):
        checkpoint = {
            'generation': self.generation,
            'single_probs': self.single_probs,
            'pair_interactions': self.pair_interactions,
            'n_mutations_probs': self.n_mutations_probs,
            'elite_memory': self.elite_memory,
            'history': history,
            'temperature': self.temperature
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str) -> List[Dict]:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.single_probs = checkpoint['single_probs']
        self.pair_interactions = checkpoint['pair_interactions']
        self.n_mutations_probs = checkpoint['n_mutations_probs']
        self.elite_memory = checkpoint['elite_memory']
        self.temperature = checkpoint['temperature']
        
        return checkpoint['history']
    
    def save_results(self, 
                     best_candidate: Candidate,
                     history: List[Dict],
                     output_dir: str = "results",
                     run_name: str = None) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        
        if run_name is None:
            run_name = f"ce_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        paths = {
            'summary': os.path.join(output_dir, f"{run_name}_summary.txt"),
            'history': os.path.join(output_dir, f"{run_name}_history.csv"),
            'candidates': os.path.join(output_dir, f"{run_name}_elite_candidates.csv"),
            'mutations': os.path.join(output_dir, f"{run_name}_mutation_analysis.csv"),
            'params': os.path.join(output_dir, f"{run_name}_parameters.json"),
        }
        
        self._save_summary(best_candidate, history, paths['summary'])
        self._save_history(history, paths['history'])
        self._save_elite_candidates(paths['candidates'])
        self._save_mutation_analysis(history, paths['mutations'])
        self._save_parameters(paths['params'])
                
        print(f"\nResults saved to {output_dir}/", flush=True)
        for key, path in paths.items():
            print(f"  {key}: {os.path.basename(path)}", flush=True)
        
        return paths
    
    def _save_summary(self, best_candidate: Candidate, history: List[Dict], filepath: str):
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CROSS-ENTROPY OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("BEST VARIANT\n")
            f.write("-"*40 + "\n")
            f.write(f"Fitness: {best_candidate.fitness:.4f}\n")
            f.write(f"Number of mutations: {len(best_candidate.mutations)}\n")
            
            mutations_list = sorted(list(best_candidate.mutations), 
                                  key=lambda m: m.position)
            mutations_str = ", ".join([str(m) for m in mutations_list])
            f.write(f"Mutations: {mutations_str}\n\n")
            
            f.write("Sequence (mutations in lowercase):\n")
            seq_list = list(self.reference_seq)
            for mut in best_candidate.mutations:
                seq_list[mut.position] = mut.new_aa.lower()
            
            seq = ''.join(seq_list)
            for i in range(0, len(seq), 60):
                f.write(f"{i+1:4d} {seq[i:i+60]}\n")
            
            f.write("\n\nOPTIMIZATION STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total generations: {len(history)}\n")
            f.write(f"Total candidates evaluated: {sum(h.get('n_evaluated', self.population_size) for h in history)}\n")
            
            f.write(f"\nFitness progression:\n")
            f.write(f"  Generation 1: {history[0]['max_fitness']:.4f}\n")
            if len(history) > 1:
                f.write(f"  Generation {len(history)}: {history[-1]['max_fitness']:.4f}\n")
            f.write(f"  Improvement: {best_candidate.fitness - history[0]['max_fitness']:.4f}\n")
            
            f.write("\n\nTOP MUTATIONS IN FINAL ELITE\n")
            f.write("-"*40 + "\n")
            mutation_counts = defaultdict(int)
            for candidate in self.elite_memory[:20]:
                for mut in candidate.mutations:
                    mutation_counts[str(mut)] += 1
            
            sorted_mutations = sorted(mutation_counts.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:10]
            for mut_str, count in sorted_mutations:
                f.write(f"  {mut_str}: {count}/20 ({count/20*100:.0f}%)\n")
    
    def _save_history(self, history: List[Dict], filepath: str):
        history_data = []
        for h in history:
            history_data.append({
                'generation': h['generation'],
                'mean_fitness': h['mean_fitness'],
                'max_fitness': h['max_fitness'],
                'min_fitness': h['min_fitness'],
                'std_fitness': h['std_fitness'],
                'elite_mean': h['elite_mean'],
                'elite_std': h['elite_std'],
                'temperature': h['temperature'],
                'best_mutations': str(h['best_candidate'].mutations),
                'best_n_mutations': len(h['best_candidate'].mutations)
            })
        
        df = pd.DataFrame(history_data)
        df.to_csv(filepath, index=False)
    
    def _save_elite_candidates(self, filepath: str):
        elite_data = []
        for i, candidate in enumerate(self.elite_memory):
            mutations_list = sorted(list(candidate.mutations), 
                                  key=lambda m: m.position)
            mutations_str = ", ".join([str(m) for m in mutations_list])
            
            elite_data.append({
                'rank': i + 1,
                'fitness': candidate.fitness,
                'n_mutations': len(candidate.mutations),
                'mutations': mutations_str,
                'sequence': candidate.sequence
            })
        
        df = pd.DataFrame(elite_data)
        df.to_csv(filepath, index=False)
    
    def _save_mutation_analysis(self, history: List[Dict], filepath: str):
        all_mutations = defaultdict(list)
        position_stats = defaultdict(lambda: {'count': 0, 'fitness_sum': 0})
        aa_changes = defaultdict(lambda: {'count': 0, 'fitness_sum': 0})
        
        for candidate in self.elite_memory:
            for mut in candidate.mutations:
                mut_str = str(mut)
                all_mutations[mut_str].append(candidate.fitness)
                
                position_stats[mut.position]['count'] += 1
                position_stats[mut.position]['fitness_sum'] += candidate.fitness
                
                aa_change = f"{mut.original_aa}->{mut.new_aa}"
                aa_changes[aa_change]['count'] += 1
                aa_changes[aa_change]['fitness_sum'] += candidate.fitness
        
        mutation_data = []
        for mut_str, fitness_list in all_mutations.items():
            mutation_data.append({
                'mutation': mut_str,
                'count': len(fitness_list),
                'mean_fitness': np.mean(fitness_list),
                'std_fitness': np.std(fitness_list),
                'max_fitness': np.max(fitness_list),
                'min_fitness': np.min(fitness_list)
            })
        
        mutation_data.sort(key=lambda x: (-x['count'], -x['mean_fitness']))
        
        df = pd.DataFrame(mutation_data)
        df.to_csv(filepath, index=False)
    
    def _save_parameters(self, filepath: str):
        params = {
            'target_positions': self.target_positions,
            'population_size': self.population_size,
            'elite_fraction': self.elite_size / self.population_size,
            'elite_size': self.elite_size,
            'smoothing_alpha': self.smoothing_alpha,
            'max_mutations': self.max_mutations,
            'initial_temperature': 1.0,
            'final_temperature': self.temperature,
            'epistasis_threshold': self.epistasis_threshold,
            'device': self.device,
            'n_positions': self.n_positions,
            'reference_length': len(self.reference_seq)
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    def export_top_variants(self, 
                           n_variants: int = 100,
                           output_file: str = "top_variants.fasta") -> str:
        with open(output_file, 'w') as f:
            f.write(">Wild_type_GFP\n")
            f.write(f"{self.reference_seq}\n")
            
            for i, candidate in enumerate(self.elite_memory[:n_variants]):
                mutations_list = sorted(list(candidate.mutations), 
                                      key=lambda m: m.position)
                mutations_str = "_".join([str(m) for m in mutations_list])
                
                f.write(f">Variant_{i+1}_fitness_{candidate.fitness:.4f}_{mutations_str}\n")
                f.write(f"{candidate.sequence}\n")
        
        print(f"Exported {min(n_variants, len(self.elite_memory))} variants to {output_file}", flush=True)
        return output_file


def run_multi_start_optimization(gleam_predictor,
                                target_positions: List[int],
                                n_starts: int = 3,
                                n_generations: int = 50,
                                save_results: bool = True,
                                output_dir: str = "results",
                                **kwargs) -> Tuple[Candidate, List[Dict]]:
    best_overall = None
    best_fitness = -np.inf
    all_histories = []
    best_optimizer = None
    
    for start in range(n_starts):
        print(f"\n{'='*50}", flush=True)
        print(f"Starting run {start + 1}/{n_starts}", flush=True)
        print(f"{'='*50}", flush=True)
        
        np.random.seed(42 + start)
        
        init_keys = ["population_size", "elite_fraction", "max_mutations", 
                     "epistasis_threshold", "reference_sequence", "device"]
        constructor_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}

        optimizer = EpistaticCrossEntropy(
            gleam_predictor,
            target_positions,
            smoothing_alpha=0.1 + 0.1 * np.random.randn(),
            temperature=2.0 + 0.2 * np.random.randn(),
            **constructor_kwargs
        )
        
        if start == 0:
            optimizer.n_mutations_probs = np.array([0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
            optimizer.n_mutations_probs /= optimizer.n_mutations_probs[1:].sum()
        elif start == 1:
            optimizer.n_mutations_probs = np.array([0.0, 0.0, 0.2, 0.3, 0.3, 0.2, 0.0])
            optimizer.n_mutations_probs /= optimizer.n_mutations_probs[1:].sum()
        else:
            optimizer.n_mutations_probs = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4])
            optimizer.n_mutations_probs /= optimizer.n_mutations_probs[1:].sum()
        
        optimize_keys = ["batch_size", "local_search_elite", "subsample_size_elite", 
                         "subsample_size_final", "adaptive_temperature", "save_checkpoint", "verbose"]
        optimize_kwargs = {k: v for k, v in kwargs.items() if k in optimize_keys}

        best_candidate, history = optimizer.optimize(
            n_generations=n_generations,
            checkpoint_path=f"ce_checkpoint_start{start}.pkl",
            **optimize_kwargs
        )
        
        all_histories.append(history)
        
        if best_candidate.fitness > best_fitness:
            best_overall = best_candidate
            best_fitness = best_candidate.fitness
            best_optimizer = optimizer
        
    if save_results and best_optimizer:
        print(f"\n{'='*50}", flush=True)
        print("SAVING RESULTS", flush=True)
        print(f"{'='*50}", flush=True)
        
        paths = best_optimizer.save_results(
            best_overall,
            all_histories[0],
            output_dir=output_dir
        )
        
        fasta_path = best_optimizer.export_top_variants(
            n_variants=100,
            output_file=os.path.join(output_dir, "top_variants.fasta")
        )
        
        all_histories_data = []
        for start_idx, history in enumerate(all_histories):
            for h in history:
                h_copy = h.copy()
                h_copy['start_run'] = start_idx + 1
                all_histories_data.append({
                    'start_run': start_idx + 1,
                    'generation': h['generation'],
                    'max_fitness': h['max_fitness'],
                    'elite_mean': h['elite_mean']
                })
        
        multi_start_df = pd.DataFrame(all_histories_data)
        multi_start_path = os.path.join(output_dir, "multi_start_comparison.csv")
        multi_start_df.to_csv(multi_start_path, index=False)
        print(f"Multi-start comparison saved to: {multi_start_path}", flush=True)
    
    return best_overall, all_histories


if __name__ == "__main__":    
    gleam_model = GLEAM(
        model_path=..., # Path to gleam.pt
        device=... # 'cuda' or 'cpu' or None for auto detection
    ) 
    
    target_positions = [
        13, 15, 17, 41, 43, 45, 60, 
        61, 63, 64, 67, 68, 71, 91, 
        93, 95, 107, 109, 111, 118, 
        120, 122, 144, 145, 147, 149, 
        162, 164, 166, 157, 180, 182, 
        184, 200, 202, 204, 219, 221, 223
    ]
    
    best_variant, histories = run_multi_start_optimization(
        gleam_model,
        target_positions,
        n_starts=2,
        n_generations=10,
        save_results=True,
        output_dir=..., # Path to output dir
        population_size=1024,
        elite_fraction=0.2,
        batch_size=64,
        local_search_elite=0,
        subsample_size_elite=0,
        subsample_size_final=340,
        save_checkpoint=False,
        verbose=True
    )
    
    print(f"\n{'='*50}", flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Best variant fitness: {best_variant.fitness:.4f}", flush=True)
    print(f"Mutations: {best_variant.mutations}", flush=True)
    print(f"Sequence: {best_variant.sequence}", flush=True)