## Overview

This project presents a comprehensive machine learning pipeline for optimizing the fluorescence brightness of Aequorea victoria Green Fluorescent Protein (avGFP) through directed evolution with constrained mutations. The methodology combines deep learning-based fitness prediction with evolutionary optimization algorithms to discover high-brightness protein variants within a maximum of 6 mutations from the wild-type sequence.

The optimization pipeline consists of three main components working in synergy: **GLEAM** (Global-Local Embeddings Attention Model) serves as a neural oracle for predicting fluorescence brightness from amino acid sequences, **Machine Learning Directed Evolution (MLDE)** employs corpus-based masking strategies with large language models for mutation sampling, and **Cross-Entropy Monte Carlo** optimization leverages probabilistic modeling of epistatic interactions for efficient exploration of the mutation landscape.

The fundamental challenge addressed is the astronomical size of the protein sequence space. For a protein of length $L$ with 20 possible amino acids, the total sequence space is $20^L$. Even with the constraint of maximum 6 mutations, the combinatorial space remains intractable for exhaustive search:

$$\sum_{k=1}^{6} \binom{L}{k} \cdot 19^k$$

where $L = 238$ for avGFP, yielding approximately $10^{13}$ possible variants. Our methodology theorethically navigates efficiently this vast space by learning from limited experimental data and exploiting structural and evolutionary patterns encoded in protein language models.

In addition to the task of fluorescence, the task was to maintain sufficient thermal stability of the protein when mutations were introduced into it. 
Thermostability of protein molecules characterizes their ability to preserve their native spatial structure and biological activity when exposed to elevated temperatures. 

From the biochemical point of view, this property is due to the stability of secondary, tertiary and quaternary structures to thermal denaturation, which is accompanied by the destruction of noncovalent interactions. These include hydrogen bonds, ionic and hydrophobic interactions, and vanderwaals forces. The amino acid composition plays a special role: the increased content of hydrophobic residues in the protein core, the presence of disulfide bridges and salt bridges (ion pairs) between charged amino acid residues contributes to an increase in the activation energy of the denaturation process.

From a biophysical position, thermostability is determined by the free energy of stabilization of the native conformation (ΔG), which is described by the equation:

$$\Delta G = \Delta H - T \Delta S$$

where ΔH is the enthalpy change, T is the absolute temperature, and ΔS is the entropy change of the system. Thermostable proteins have higher ΔG values than their mesophilic counterparts, reflecting their increased resistance to unfolding. The key factors affecting ΔG are the packing density of the hydrophobic core, which reduces the entropic contribution of the unfolded state (ΔS), and the enhancement of enthalpic interactions (ΔH), such as the formation of additional hydrogen bonds and ion pairs.

For reproducibility, Conda package manager was utiziled. We strongly recommend to use it for evaluating purposes:

```bash
conda env create -f environment.yml
conda activate synbio
```

## GLEAM

### Implementation

GLEAM (Global-Local Embeddings Attention Model) represents a neural architecture specifically designed for predicting protein fitness from sequence information. The model addresses the fundamental challenge of capturing both global protein context and local mutational effects through a attention mechanism that integrates features at multiple scales.

The architecture begins with ESM-2 (Evolutionary Scale Modeling) with 3B params, a transformer-based protein language model pre-trained on millions of protein sequences. ESM-2 provides rich contextual embeddings $\mathbf{h}_i \in \mathbb{R}^{d}$ for each amino acid position $i$, where $d = 2560$ represents the embedding dimension. These embeddings capture evolutionary relationships and structural constraints learned from massive protein databases.

GLEAM processes these embeddings through two parallel pathways. The global pathway computes a sequence-level representation by averaging token embeddings and passing them through dense layers:

$$\mathbf{g} = \text{FFN}_{\text{global}}\left(\frac{1}{L}\sum_{i=1}^{L} \mathbf{h}_i\right)$$

The local pathway focuses on mutation neighborhoods by extracting windows of size $w$ around each mutated position. For a mutation at position $j$, the local context is defined as:

$$\mathbf{L}_j = [\mathbf{h}_{j-w/2}, \ldots, \mathbf{h}_{j+w/2}] \in \mathbb{R}^{w \times d}$$

Local transformer blocks process these windows using multi-head self-attention with spectral normalization for training stability:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are query, key, and value matrices derived from spectral-normalized linear projections.

Cross-attention mechanism treats global features as queries and local mutation features as keys and values:

$$\mathbf{c} = \text{CrossAttention}(\mathbf{g}, \mathbf{L}_{\text{mutations}})$$

This design enables the model to consider how local mutations interact with the global protein structure and function, capturing epistatic effects that are crucial for accurate fitness prediction.

The final prediction incorporates both brightness estimation and uncertainty quantification through Monte Carlo dropout. The model outputs a brightness score $\hat{y}$ and an uncertainty estimate $\sigma^2$:

$$\hat{y}, \sigma^2 = \text{Head}([\mathbf{c}; \mathbf{g}])$$

Training employs a combined loss function that balances multiple objectives:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \alpha \mathcal{L}_{\text{Huber}} + \beta \mathcal{L}_{\text{NLL}} + \gamma |\theta|_2^2$$

where $\mathcal{L}_{\text{MSE}}$ ensures accurate mean prediction, $\mathcal{L}_{\text{Huber}}$ provides robustness to outliers, $\mathcal{L}_{\text{NLL}}$ enforces well-calibrated uncertainty estimates, and the L2 term prevents overfitting.

### Usage

The GLEAM model can be loaded and used for fitness prediction through a straightforward interface:

```python
from gleam import GLEAM

# Initialize the predictor
predictor = GLEAM(
    model_path="path/to/gleam.pt",
    esm_model_name="esm2_t36_3B_UR50D",
    device="cuda",  # or "cpu"
    window_size=7, # Should not be changed
    max_mutations=6 # Should not be changed
)

# Single sequence prediction
sequence = "MSKGEELFT…"  # Your mutated GFP sequence
result = predictor.predict_single(
    sequence=sequence,
    return_uncertainty=True,
    mc_samples=10
)

# Access results
brightness = result['predicted_brightness']
uncertainty = result['uncertainty']
mutations = result['mutations']  # List of (original_aa, position, new_aa)

# Batch prediction for multiple sequences
sequences = ["sequence1", "sequence2", …]
batch_results = predictor.predict_batch(
    sequences=sequences,
    return_uncertainty=True,
    batch_size=16
)

# DataFrame integration
import pandas as pd
df = pd.DataFrame({'sequence': sequences})
result_df = predictor.predict_from_dataframe(
    df=df,
    sequence_column='sequence'
)
```

## Machine Learning Directed Evolution

### Implementation

Machine Learning Directed Evolution (MLDE) is approach for protein optimization that mimics natural evolutionary processes. Unlike traditional directed evolution that relies on random mutagenesis, MLDE employs learned patterns from protein language models to guide mutation generation toward promising regions of sequence space.

The MLDE algorithm operates through iterative cycles of mutation, evaluation, and selection, similar to natural evolution but with bias toward beneficial changes. The process begins with a population of protein sequences and applies a masking strategy informed by sequence statistics and importance measures derived from corpus analysis.

Central to MLDE is the concept of k-mer importance weighting, which identifies sequence regions most suitable for modification. The algorithm analyzes the frequency and distribution of k-mers (subsequences of length k) across a corpus of known sequences to compute TF-IDF (Term Frequency-Inverse Document Frequency) scores:

$$\text{TF-IDF}(k\text{-mer}) = \frac{f_{k\text{-mer},\text{seq}}}{|S|} \cdot \log\left(\frac{N}{N_{k\text{-mer}}}\right)$$

where $f_{k\text{-mer},\text{seq}}$ is the frequency of the k-mer in the sequence, $|S|$ is the total number of k-mers in the sequence, $N$ is the total number of sequences in the corpus, and $N_{k\text{-mer}}$ is the number of sequences containing the k-mer.

Additionally, the algorithm computes entropy scores for each k-mer to measure their information content:

$$H(k\text{-mer}) = -p_{k\text{-mer}} \log p_{k\text{-mer}}$$

where $p_{k\text{-mer}}$ is the probability of observing the k-mer in the corpus. High entropy k-mers represent rare or unusual sequence patterns that may be good candidates for modification.

The importance score for each position combines normalized TF-IDF and entropy measures:

$$I_{\text{pos}} = \frac{\text{TF-IDF}_{\text{norm}} + H_{\text{norm}}}{2}$$

MLDE employs two complementary masking strategies. Random masking selects positions uniformly to maintain exploration, while importance-based masking targets positions with low importance scores, hypothesizing that modifying less conserved regions is more likely to yield beneficial variants.

The masked positions are then filled using ESM-2's masked language modeling capabilities. Given a sequence with masked tokens, ESM-2 predicts the most likely amino acids based on evolutionary patterns learned from millions of protein sequences:

$$p(\text{AA}|\text{context}) = \text{softmax}(\text{ESM-2}(\text{masked sequence}))$$

This approach leverages the vast evolutionary knowledge embedded in the language model to suggest mutations that are both structurally plausible and evolutionarily reasonable.

Each generated variant is evaluated using GLEAM to obtain fitness scores, and the population undergoes selection pressure to retain high-fitness individuals. The iterative nature of MLDE allows the algorithm to build upon successful mutations while exploring new combinations, leading to progressive improvement in population fitness.

### Usage

The Machine Learning Directed Evolution optimizer can be configured and executed as follows:

```python
from gleam import GLEAM
from mlde_optimizer import MLDEOptimizer

# Initialize GLEAM as the fitness oracle
gleam_model = GLEAM(model_path="path/to/gleam.pt", device="cuda")

# Wild-type avGFP sequence
wt_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK…"

# Initialize MLDE optimizer
optimizer = MLDEOptimizer(
    gleam_model=gleam_model,
    wt_sequence=wt_sequence,
    max_mutations=6,
    k_mer_size=1  # Size of sequence fragments for analysis
)

# Run optimization
best_variants = optimizer.optimize(
    population_size=128,    # Number of candidates per generation
    beam_size=4,           # Expansion factor for beam search
    num_iterations=100,    # Maximum generations
    random_ratio=0.6,      # Fraction using random vs. importance masking
    early_stopping_patience=10
)

# Access results
for variant in best_variants[:5]:  # Top 5 variants
    print(f"Mutations: {variant['mutation_string']}")
    print(f"Predicted brightness: {variant['brightness']:.4f}")
    print(f"Sequence: {variant['sequence']}")

# Export all results
results_df = optimizer.get_all_results_df()
results_df.to_csv("mlde_results.csv", index=False)
```

## Cross-Entropy

### Implementation

The Cross-Entropy method represents a powerful Monte Carlo optimization technique adapted for protein engineering. Unlike gradient-based methods that can become trapped in local optima, the cross-entropy approach maintains probabilistic distributions over the solution space and iteratively refines these distributions based on elite samples.

In the context of protein optimization, the cross-entropy method models the probability of selecting specific amino acids at each target position. The algorithm maintains a probability distribution $\theta_{i,a}$ representing the likelihood of choosing amino acid $a$ at position $i$:

$$P(\text{AA}=a \text{ at position } i) = \theta_{i,a}$$

subject to the constraint $\sum_{a} \theta_{i,a} = 1$ for all positions $i$.

The optimization proceeds through iterative cycles of sampling, evaluation, and distribution updating. In each generation, the algorithm samples $N$ candidate solutions from the current distribution:

$$\mathbf{x}^{(j)} \sim P(\mathbf{x}|\boldsymbol{\theta})$$

where $\mathbf{x}^{(j)}$ represents the $j$-th sampled protein variant.

After evaluating all candidates using GLEAM, the algorithm selects the top $\rho \cdot N$ elite samples based on fitness scores, where $\rho$ is the elite fraction (typically 0.1-0.3). The distribution parameters are then updated using maximum likelihood estimation on the elite set:

$$\theta_{i,a}^{(t+1)} = \frac{\sum_{j \in \text{Elite}} \mathbf{1}[\mathbf{x}^{(j)}_i = a]}{|\text{Elite}|}$$

To prevent premature convergence and maintain exploration, the algorithm applies smoothing:

$$\theta_{i,a}^{(t+1)} = \alpha \theta_{i,a}^{(t+1)} + (1-\alpha) \theta_{i,a}^{(t)}$$

where $\alpha \in [0,1]$ controls the balance between learning from new data and retaining previous knowledge.

Our approach utilises explicit modeling of epistatic interactions between positions. The algorithm maintains pairwise interaction terms $W_{i,a,j,b}$ that capture the synergistic or antagonistic effects of having amino acid $a$ at position $i$ and amino acid $b$ at position $j$:

$$P(\text{AA}=a \text{ at } i | \text{mutations at other positions}) \propto \exp\left(\beta \sum_{j \neq i} W_{i,a,j,b_j}\right)$$

where $\beta$ is a temperature parameter controlling the strength of epistatic effects.

These interaction terms are learned through co-occurrence analysis in elite populations. The epistatic score between amino acid pairs is computed as:

$$W_{i,a,j,b} = \log\left(\frac{O_{i,a,j,b} + \epsilon}{E_{i,a,j,b} + \epsilon}\right)$$

where $O_{i,a,j,b}$ is the observed co-occurrence frequency and $E_{i,a,j,b}$ is the expected frequency under independence assumption.

The algorithm incorporates advanced sampling techniques including Gibbs sampling for generating candidates with epistatic considerations:

$$P(\text{mutation at } i | \text{other mutations}) \propto \exp\left(\frac{\log \theta_{i,a} + \sum_j W_{i,a,j,b_j}}{T}\right)$$

where $T$ is a temperature parameter that controls exploration vs. exploitation balance.

Local search enhancement is performed on elite candidates through systematic neighborhood exploration, considering single-point mutations, additions, deletions, and amino acid substitutions. This hybrid approach combines the global search capabilities of cross-entropy sampling with the fine-tuning power of local optimization.

The cross-entropy method naturally handles the constraint of maximum mutations through controlled sampling from the mutation number distribution, ensuring that generated variants remain within the specified bounds while maximizing the probability of discovering high-fitness combinations.

### Usage

The Epistatic Cross-Entropy optimizer can be configured and executed as follows:

```python
from cross_entropy_optimizer import EpistaticCrossEntropy, run_multi_start_optimization

# Define target positions for optimization (key residues in GFP)
target_positions = [13, 15, 17, 41, 43, 45, 60, 61, 63, 64, 67, 68, 71, 
                   91, 93, 95, 107, 109, 111, 118, 120, 122, 144, 145]

# Initialize optimizer
optimizer = EpistaticCrossEntropy(
    gleam_predictor=gleam_model,
    target_positions=target_positions,
    population_size=1024,
    elite_fraction=0.2,
    max_mutations=6,
    temperature=1.0,
    epistasis_threshold=0.1
)

# Single optimization run
best_candidate, history = optimizer.optimize(
    n_generations=50,
    batch_size=64,
    local_search_elite=10,
    adaptive_temperature=True,
    save_checkpoint=True
)

# Multi-start optimization for robustness
best_variant, all_histories = run_multi_start_optimization(
    gleam_predictor=gleam_model,
    target_positions=target_positions,
    n_starts=3,
    n_generations=50,
    population_size=1024,
    save_results=True,
    output_dir="results"
)

# Access final results
print(f"Best fitness: {best_variant.fitness:.4f}")
print(f"Mutations: {best_variant.mutations}")
print(f"Sequence: {best_variant.sequence}")

# Export top variants
optimizer.export_top_variants(
    n_variants=100,
    output_file="top_variants.fasta"
)
```

## Thermostability analysis

### Implementation

To predict the thermostability of GFP mutants, we used TemBERTure<sub>tm</sub>, a deep learning package for protein thermostability prediction. It consists of three components: TemBERTureDB, a large-curated database of thermophilic and non-thermophilic sequences, TemBERTure<sub>cls</sub>, a classifier, and TemBERTure<sub>tm</sub>, a regression model, which predicts, respectively, the thermal class (non-thermophilic or thermophilic) and melting temperature of a protein, based on its primary sequence. Both models are built upon the existing protBERT-BFD language model and fine-tuned through an adapter-based approach.

The TemBERTure<sub>cls</sub> model architecture was based on the protBERT-BFD framework, with lightweight bottleneck adapter layers inserted between each transformer layer. The model takes a protein sequence as input and outputs a score indicating the classification score of the sequence being thermophilic or non-thermophilic.

TemBERTure<sub>tm</sub> is a sequence-based regression model designed to predict the protein melting temperature (T<sub>m</sub>) directly from its amino acid sequence. This model has the same underlying architecture configuration and tokenization as TemBERTure<sub>cls</sub>, with a regression head.

Both models leverage the pre-trained protBERT-BFD architecture to minimize the number of trainable parameters, employing an adapter-based fine-tuning approach for optimization.

The TemBERTure model and data are available at: https://github.com/ibmm-unibe-ch/TemBERTure

### Usage

...
