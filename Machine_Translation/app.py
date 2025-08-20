from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import requests
from model import TransformerSeq2Seq,translate
from utils import load_tokenizers_and_embeddings

import torch

   # class mô hình của bạn

app = FastAPI()

# ===== 1. Load model và tokenizer khi khởi động server =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load 1 lần khi start server =====
resources = load_tokenizers_and_embeddings()
tokenizer_vi = resources["tokenizer_vi"]
embedding_matrix_vi = resources["embedding_vi"]
tokenizer_en = resources["tokenizer_en"]
embedding_matrix_en = resources["embedding_en"]
device = resources["device"]

print("✅ Tokenizers & embeddings loaded!")
if isinstance(embedding_matrix_en, torch.Tensor):
    embed_dim = embedding_matrix_en.size(1)
else:  # nn.Embedding
    embed_dim = embedding_matrix_en.embedding_dim
max_len = 128
batch_size = 32
# Load model
model = TransformerSeq2Seq(
    embed_dim=embed_dim,
    vocab_size=tokenizer_vi.vocab_size,   # hoặc len(tokenizer_vi)
    embedding_decoder=embedding_matrix_vi,      # embedding target đã có sẵn
    num_heads=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    freeze_decoder_emb=True,
    max_len=max_len
)
MODEL_URL = "https://huggingface.co/nemabruh404/Machine_Translation/resolve/main/model_state_dict.pt"

# Fetch model từ Hub
checkpoint_bytes = BytesIO(requests.get(MODEL_URL).content)
checkpoint = torch.load(checkpoint_bytes, map_location=device)

# Load state dict
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ Model loaded from Hugging Face Hub")
print("Model loaded")
class TranslationRequest(BaseModel):
    text: str
# ===== Endpoint dịch =====
@app.post("/translate")
def translate_api(req: TranslationRequest):
    output = translate(
        model=model,
        src_sentence=req.text,
        tokenizer_src=tokenizer_en,   # tiếng Anh -> input
        tokenizer_tgt=tokenizer_vi,   # tiếng Việt -> output
        embedding_src=embedding_matrix_en,
        device=device,
        max_len=max_len
    )
    return {"input": req.text, "translation": output}