import json

input_file = "medquad_all.jsonl"
output_file = "medquad_alpaca.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        record = json.loads(line)
        q = record.get("question", "").strip()
        a = record.get("answer", "").strip()

        entry = {
            "instruction": "Answer the medical question.",
            "input": q,
            "response": a
        }
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("✅ Converted dataset to Alpaca format:", output_file)
