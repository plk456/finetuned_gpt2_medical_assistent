import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# -------------------- 1️⃣ Model & Adapter --------------------
model_name = "gpt2-large"       # MUST match training model
adapter_path = "./qlora_adapter"  # Path where you saved the adapter

# 4-bit QLoRA quantization (same as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# -------------------- 2️⃣ Load Base Model --------------------
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",   # automatically maps layers to GPU(s)
)

# -------------------- 3️⃣ Tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- 4️⃣ Load LoRA Adapter --------------------
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# -------------------- 5️⃣ Inference --------------------
prompt = (
    "User: Can you explain the benefits of meditation?\n"
    "Assistant:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.6,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
