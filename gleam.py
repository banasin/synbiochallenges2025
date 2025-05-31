import torch
import esm
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple, Optional
import warnings
import re
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
warnings.filterwarnings('ignore')

class PositionWiseFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(residual + x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.q_proj = nn.utils.spectral_norm(self.q_proj)
        self.k_proj = nn.utils.spectral_norm(self.k_proj)
        self.v_proj = nn.utils.spectral_norm(self.v_proj)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        context = context.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        context = context.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, embed_dim]
        
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return self.norm(residual + output), attn_weights

class LocalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, stochastic_depth_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PositionWiseFFN(embed_dim, ffn_dim, dropout)
        self.stochastic_depth_prob = stochastic_depth_prob
        
    def forward(self, x, mask=None, apply_stochastic_depth=True):
        apply_sd = self.training and apply_stochastic_depth and torch.rand(1).item() < self.stochastic_depth_prob
        
        if not apply_sd:
            attn_output, attn_weights = self.attention(x, mask)
            x = attn_output
        
        if not apply_sd:
            x = self.ffn(x)
        
        return x, attn_weights if not apply_sd else (x, None)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        
        residual = query
        
        query = self.norm1(query)
        key_value = self.norm2(key_value)
        
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch_size, num_heads, query_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, kv_len, head_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, query_len, head_dim]
        context = context.transpose(1, 2)  # [batch_size, query_len, num_heads, head_dim]
        context = context.reshape(batch_size, query_len, -1)  # [batch_size, query_len, embed_dim]
        
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return residual + output, attn_weights

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2, gamma=1e-5, delta=0.25):
        super().__init__()
        self.alpha = alpha  # Huber Loss
        self.beta = beta    # NLL Loss
        self.gamma = gamma  # L2 регуляризации
        self.delta = delta  # Huber loss
        self.mse = nn.MSELoss(reduction='mean')
        self.huber = nn.HuberLoss(reduction='mean', delta=delta)
        
    def forward(self, pred, target, uncertainty=None, model=None):
        mse_loss = self.mse(pred, target)
        
        huber_loss = self.huber(pred, target)
        
        nll_loss = 0.0
        if uncertainty is not None:
            uncertainty = torch.clamp(uncertainty, min=1e-6)
            nll_loss = torch.mean(0.5 * torch.log(uncertainty) + 
                                  0.5 * (pred - target)**2 / uncertainty)
        
        l2_loss = 0.0
        if model is not None:
            for param in model.parameters():
                l2_loss += torch.norm(param, 2)
        
        total_loss = mse_loss + self.alpha * huber_loss + self.beta * nll_loss + self.gamma * l2_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'huber': huber_loss.item(),
            'nll': nll_loss.item() if isinstance(nll_loss, torch.Tensor) else nll_loss,
            'l2': l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss,
            'total': total_loss.item()
        }

class GLEAM_MODULE(nn.Module): # Global-Local Embeddings Attention Model
    def __init__(
        self, 
        embed_dim=2560,                # ESM2 embeddings
        hidden_dim_global=1024,
        hidden_dim_local=512,
        num_local_transformer_layers=2,
        num_heads=4,
        dropout_rate=0.2,
        stochastic_depth_prob=0.1,
        max_mutations=6,
        window_size=7,
        predict_uncertainty=True,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predict_uncertainty = predict_uncertainty
        
        self.global_dense1 = nn.Linear(embed_dim, hidden_dim_global)
        self.global_bn1 = nn.BatchNorm1d(hidden_dim_global)
        self.global_dense2 = nn.Linear(hidden_dim_global, hidden_dim_global // 2)
        self.global_bn2 = nn.BatchNorm1d(hidden_dim_global // 2)
        
        self.local_transformer_layers = nn.ModuleList([
            LocalTransformerBlock(
                embed_dim, 
                num_heads,
                hidden_dim_local,
                dropout=dropout_rate,
                stochastic_depth_prob=stochastic_depth_prob
            )
            for _ in range(num_local_transformer_layers)
        ])
        
        self.cross_attention = CrossAttention(
            embed_dim, 
            num_heads,
            dropout=dropout_rate
        )
        
        self.global_projection = nn.Linear(hidden_dim_global // 2, embed_dim)
        
        aggregation_input_dim = embed_dim + hidden_dim_global // 2
        self.aggregation_layers = nn.Sequential(
            nn.Linear(aggregation_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        self.prediction_head = nn.Linear(64, 1)
        
        if predict_uncertainty:
            self.uncertainty_head = nn.Linear(64, 1)
            self.softplus = nn.Softplus()
    
    def forward(self, global_embed, local_embed, return_attention=False):
        batch_size = global_embed.shape[0]
        
        x_global = self.global_dense1(global_embed)
        x_global = self.global_bn1(x_global)
        x_global = F.gelu(x_global)
        x_global = F.dropout(x_global, p=0.2, training=self.training)
        
        x_global = self.global_dense2(x_global)
        x_global = self.global_bn2(x_global)
        x_global = F.gelu(x_global)
        
        x_global_projected = self.global_projection(x_global)
        x_global_expanded = x_global_projected.unsqueeze(1)
        
        # [batch_size, max_mutations, window_size, embed_dim] -> [batch_size*max_mutations, window_size, embed_dim]
        max_mutations, window_size, _ = local_embed.shape[1:]
        x_local = local_embed.view(batch_size * max_mutations, window_size, self.embed_dim)
        
        local_norms = torch.norm(x_local, dim=2)
        attention_mask = (local_norms > 1e-6).float().unsqueeze(1).unsqueeze(2)
        
        attention_weights_list = []
        
        for layer in self.local_transformer_layers:
            x_local, attn_weights = layer(x_local, attention_mask)
            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)
        
        # [batch_size*max_mutations, window_size, embed_dim] -> [batch_size, max_mutations, window_size, embed_dim]
        x_local = x_local.view(batch_size, max_mutations, window_size, self.embed_dim)
        
        # [batch_size, max_mutations, window_size, embed_dim] -> [batch_size, max_mutations, embed_dim]
        x_local = x_local.mean(dim=2)
        
        # Global (Q): [batch_size, 1, embed_dim//2]
        # Local (K/V): [batch_size, max_mutations, embed_dim]
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x_global_expanded, x_local
        )
        
        if return_attention:
            attention_weights_list.append(cross_attn_weights)
        
        # [batch_size, 1, embed_dim] -> [batch_size, embed_dim]
        cross_attn_output = cross_attn_output.squeeze(1)
        
        # [batch_size, embed_dim + embed_dim//2]
        combined = torch.cat([cross_attn_output, x_global], dim=1)
        
        x = self.aggregation_layers(combined)
        
        prediction = self.prediction_head(x).squeeze(-1)
        
        if self.predict_uncertainty:
            uncertainty = self.softplus(self.uncertainty_head(x)).squeeze(-1)
            if return_attention:
                return prediction, uncertainty, attention_weights_list
            return prediction, uncertainty
        
        if return_attention:
            return prediction, None, attention_weights_list
        return prediction, None

class GLEAM:
    def __init__(
        self, 
        model_path: str,
        esm_model_name: str = "esm2_t36_3B_UR50D",
        device: Optional[str] = None,
        window_size: int = 7,
        max_mutations: int = 238
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.window_size = window_size
        self.max_mutations = max_mutations
        
        self.reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        
        print(f"Loading ESM model {esm_model_name}...", flush=True)
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()
        
        print(f"Loading GLEAM from {model_path}...", flush=True)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.gleam_model = GLEAM_MODULE(**checkpoint['hyperparameters'])
        self.gleam_model.load_state_dict(checkpoint['model_state_dict'])
        self.gleam_model.to(self.device)
        self.gleam_model.eval()
        
        print(f"Models are loaded on: {self.device}", flush=True)
    
    def _set_dropout_train_mode(self, training: bool):
        for module in self.gleam_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.training = training
    
    def _parse_mutations(self, sequence: str) -> List[Tuple[str, int, str]]:
        mutations = []
        
        if len(sequence) != len(self.reference_sequence):
            raise ValueError(f"Sequence length ({len(sequence)}) isn't equal to reference ({len(self.reference_sequence)})")
        
        for i, (ref_aa, mut_aa) in enumerate(zip(self.reference_sequence, sequence)):
            if ref_aa != mut_aa:
                mutations.append((ref_aa, i, mut_aa))
        
        if len(mutations) > self.max_mutations:
            print(f"Warning: found {len(mutations)} mutations, but processing only first {self.max_mutations}")
            mutations = mutations[:self.max_mutations]
        
        return mutations
    
    def _get_embeddings(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_converter = self.esm_alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([("sequence", sequence)])
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
            token_representations = results["representations"][36]
            
            global_embedding = token_representations[0, 1:-1].mean(0)  # Исключаем CLS и SEP токены
            
            mutations = self._parse_mutations(sequence)
            
            local_embeddings = torch.zeros(self.max_mutations, self.window_size, token_representations.shape[-1])
            
            for mut_idx, (ref_aa, pos, mut_aa) in enumerate(mutations):
                if mut_idx >= self.max_mutations:
                    break
                    
                start_pos = max(0, pos - self.window_size // 2)
                end_pos = min(len(sequence), pos + self.window_size // 2 + 1)
                
                window_embeddings = token_representations[0, start_pos + 1:end_pos + 1]
                
                actual_window_size = window_embeddings.shape[0]
                if actual_window_size <= self.window_size:
                    local_embeddings[mut_idx, :actual_window_size] = window_embeddings
                else:
                    local_embeddings[mut_idx] = window_embeddings[:self.window_size]
        
        return global_embedding.cpu(), local_embeddings.cpu()
    
    def _get_embeddings_batch(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_converter = self.esm_alphabet.get_batch_converter()
        batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)

        batch_size = len(sequences)
        embed_dim = 2560

        global_embeddings = torch.zeros(batch_size, embed_dim)
        local_embeddings = torch.zeros(batch_size, self.max_mutations, self.window_size, embed_dim)

        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
            token_representations = results["representations"][36]

            for i, sequence in enumerate(sequences):
                seq_length = len(sequence)
                global_embeddings[i] = token_representations[i, 1:seq_length+1].mean(0)

                mutations = self._parse_mutations(sequence)

                for mut_idx, (ref_aa, pos, mut_aa) in enumerate(mutations):
                    if mut_idx >= self.max_mutations:
                        break

                    start_pos = max(0, pos - self.window_size // 2)
                    end_pos = min(len(sequence), pos + self.window_size // 2 + 1)

                    window_embeddings = token_representations[i, start_pos + 1:end_pos + 1]

                    actual_window_size = window_embeddings.shape[0]
                    if actual_window_size <= self.window_size:
                        local_embeddings[i, mut_idx, :actual_window_size] = window_embeddings
                    else:
                        local_embeddings[i, mut_idx] = window_embeddings[:self.window_size]

        return global_embeddings.cpu(), local_embeddings.cpu()
    
    def predict_single(
        self, 
        sequence: str, 
        return_uncertainty: bool = True,
        mc_samples: int = 10
    ) -> Dict[str, Union[float, List[Tuple[str, int, str]]]]:
        sequence = sequence.upper().strip()
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence)
        
        global_embed, local_embed = self._get_embeddings(sequence)
        
        global_embed = global_embed.unsqueeze(0).to(self.device)  # [1, embed_dim]
        local_embed = local_embed.unsqueeze(0).to(self.device)    # [1, max_mutations, window_size, embed_dim]
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            if return_uncertainty and mc_samples > 1:
                self._set_dropout_train_mode(True)
                
                for _ in range(mc_samples):
                    pred, unc = self.gleam_model(global_embed, local_embed)
                    predictions.append(pred.cpu().numpy()[0])
                    if unc is not None:
                        uncertainties.append(unc.cpu().numpy()[0])
                
                self._set_dropout_train_mode(False)
                
                mean_prediction = np.mean(predictions)
                std_prediction = np.std(predictions)
                
                if uncertainties:
                    mean_uncertainty = np.mean(uncertainties)
                else:
                    mean_uncertainty = std_prediction
                    
            else:
                pred, unc = self.gleam_model(global_embed, local_embed)
                mean_prediction = pred.cpu().numpy()[0]
                std_prediction = 0.0
                mean_uncertainty = unc.cpu().numpy()[0] if unc is not None else 0.0
        
        mutations = self._parse_mutations(sequence)
        
        result = {
            'sequence': sequence,
            'predicted_brightness': float(mean_prediction),
            'mutations': mutations,
            'num_mutations': len(mutations)
        }
        
        if return_uncertainty:
            result['uncertainty'] = float(mean_uncertainty)
            result['prediction_std'] = float(std_prediction)
        
        return result
    
    def predict_batch(
        self, 
        sequences: List[str], 
        return_uncertainty: bool = True,
        mc_samples: int = 10,
        batch_size: int = 16
    ) -> List[Dict[str, Union[float, List[Tuple[str, int, str]]]]]:
        clean_sequences = []
        for seq in sequences:
            seq = seq.upper().strip()
            seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
            clean_sequences.append(seq)

        all_results = []

        for i in tqdm(range(0, len(clean_sequences), batch_size), desc="Processing batches"):
            esm_batch_sequences = clean_sequences[i:i + batch_size]

            try:
                global_embeds, local_embeds = self._get_embeddings_batch(esm_batch_sequences)

                for j in range(0, len(esm_batch_sequences), batch_size):
                    gleam_batch_sequences = esm_batch_sequences[j:j + batch_size]
                    current_batch_size = len(gleam_batch_sequences)

                    batch_global = global_embeds[j:j + current_batch_size].to(self.device)
                    batch_local = local_embeds[j:j + current_batch_size].to(self.device)

                    batch_results = []

                    if return_uncertainty and mc_samples > 1:
                        batch_predictions = []
                        batch_uncertainties = []

                        self._set_dropout_train_mode(True)

                        for _ in range(mc_samples):
                            with torch.no_grad():
                                pred, unc = self.gleam_model(batch_global, batch_local)
                                batch_predictions.append(pred.cpu().numpy())
                                if unc is not None:
                                    batch_uncertainties.append(unc.cpu().numpy())

                        self._set_dropout_train_mode(False)

                        batch_predictions = np.array(batch_predictions)  # [mc_samples, batch_size]
                        mean_predictions = np.mean(batch_predictions, axis=0)
                        std_predictions = np.std(batch_predictions, axis=0)

                        if batch_uncertainties:
                            batch_uncertainties = np.array(batch_uncertainties)
                            mean_uncertainties = np.mean(batch_uncertainties, axis=0)
                        else:
                            mean_uncertainties = std_predictions

                    else:
                        with torch.no_grad():
                            pred, unc = self.gleam_model(batch_global, batch_local)
                            mean_predictions = pred.cpu().numpy()
                            std_predictions = np.zeros_like(mean_predictions)
                            mean_uncertainties = unc.cpu().numpy() if unc is not None else np.zeros_like(mean_predictions)

                    for k, sequence in enumerate(gleam_batch_sequences):
                        mutations = self._parse_mutations(sequence)

                        result = {
                            'sequence': sequence,
                            'predicted_brightness': float(mean_predictions[k]),
                            'mutations': mutations,
                            'num_mutations': len(mutations)
                        }

                        if return_uncertainty:
                            result['uncertainty'] = float(mean_uncertainties[k])
                            result['prediction_std'] = float(std_predictions[k])

                        batch_results.append(result)

                    all_results.extend(batch_results)
                
            except Exception as e:
                print(f"Error in {i}-{i+batch_size} batches: {e}")
                for seq in esm_batch_sequences:
                    all_results.append({
                        'sequence': seq,
                        'predicted_brightness': None,
                        'error': str(e)
                    })
        
        torch.cuda.empty_cache()
        
        return all_results
    
    def predict_from_dataframe(
        self, 
        df: pd.DataFrame, 
        sequence_column: str = 'sequence',
        return_uncertainty: bool = True,
        mc_samples: int = 10,
        batch_size: int = 16
    ) -> pd.DataFrame:
        sequences = df[sequence_column].tolist()
        results = self.predict_batch(sequences, return_uncertainty, mc_samples, batch_size)
        
        result_df = df.copy()
        
        result_df['predicted_brightness'] = [r.get('predicted_brightness') for r in results]
        result_df['num_mutations'] = [r.get('num_mutations') for r in results]
        
        if return_uncertainty:
            result_df['uncertainty'] = [r.get('uncertainty') for r in results]
            result_df['prediction_std'] = [r.get('prediction_std') for r in results]
        
        mutations_str = []
        for r in results:
            if 'mutations' in r and r['mutations']:
                mut_strs = [f"{m[0]}{m[1]+1}{m[2]}" for m in r['mutations']]
                mutations_str.append(', '.join(mut_strs))
            else:
                mutations_str.append('')
        
        result_df['mutations'] = mutations_str
        
        return result_df