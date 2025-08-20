import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizers_and_embeddings():
    # ===== Vietnamese PhoBERT =====
    tokenizer_vi = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model_vi = AutoModel.from_pretrained("vinai/phobert-base").to(device)
    embedding_matrix_vi = model_vi.embeddings.word_embeddings.weight

    # ===== English BERT =====
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model_en = AutoModel.from_pretrained("bert-base-cased-finetuned-mrpc").to(device)
    embedding_matrix_en = model_en.embeddings.word_embeddings.weight
    
    return {
        "tokenizer_vi": tokenizer_vi,
        "embedding_vi": embedding_matrix_vi,
        "tokenizer_en": tokenizer_en,
        "embedding_en": embedding_matrix_en,
        "device": device
    }
