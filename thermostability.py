import pandas as pd
import numpy as np
from temBERTure import TemBERTure
import torch

def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def initialize_models():
    print("Initializing models...", flush=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    
    print("Loading TemBERTureCLS...", flush=True)
    model_cls = TemBERTure(
        adapter_path='./temBERTure_CLS/',
        device=device,
        batch_size=1,
        task='classification'
    )
    
    print("Loading TemBERTureTM реплик...", flush=True)
    model_tm_r1 = TemBERTure(
        adapter_path='./temBERTure_TM/replica1/',
        device=device,
        batch_size=16,
        task='regression'
    )
    model_tm_r2 = TemBERTure(
        adapter_path='./temBERTure_TM/replica2/',
        device=device,
        batch_size=16,
        task='regression'
    )
    model_tm_r3 = TemBERTure(
        adapter_path='./temBERTure_TM/replica3/',
        device=device,
        batch_size=16,
        task='regression'
    )
    
    return model_cls, model_tm_r1, model_tm_r2, model_tm_r3

def process_sequences(df, model_cls, model_tm_r1, model_tm_r2, model_tm_r3):
    cls_predictions = []
    cls_scores = []
    tm_replica1_predictions = []
    tm_replica2_predictions = []
    tm_replica3_predictions = []
    
    sequences = df['sequence'].tolist()
    
    print("Processing classification...", flush=True)
    for i, seq in enumerate(sequences):
        result = model_cls.predict(seq)
        cls_predictions.append(result[0][0])
        cls_scores.append(float(result[1][0]))
    batch_size = 16
    n_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size != 0 else 0)
    
    print("Processing regression - replica 1...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        predictions = model_tm_r1.predict(batch_sequences)
        tm_replica1_predictions.extend(predictions)
        
    print("Processing regression - replica 2...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        predictions = model_tm_r2.predict(batch_sequences)
        tm_replica2_predictions.extend(predictions)
        
    print("Processing regression - replica 3...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        predictions = model_tm_r3.predict(batch_sequences)
        tm_replica3_predictions.extend(predictions)

    df_result = df.copy()
    df_result['cls_prediction'] = cls_predictions
    df_result['cls_score'] = cls_scores
    df_result['tm_replica1'] = tm_replica1_predictions
    df_result['tm_replica2'] = tm_replica2_predictions
    df_result['tm_replica3'] = tm_replica3_predictions
    df_result['tm_mean'] = df_result[['tm_replica1', 'tm_replica2', 'tm_replica3']].mean(axis=1)
    df_result['tm_std'] = df_result[['tm_replica1', 'tm_replica2', 'tm_replica3']].std(axis=1)

    return df_result

if __name__ == "__main__":
    input_csv = ... # Path to input csv with target column "sequence"
    output_csv = ... # Path to output
    
    df = load_and_process_data(input_csv)

    model_cls, model_tm_r1, model_tm_r2, model_tm_r3 = initialize_models()

    df_result = process_sequences(df, model_cls, model_tm_r1, model_tm_r2, model_tm_r3)

    print(f"Saving results to {output_csv}...")
    df_result.to_csv(output_csv, index=False)
