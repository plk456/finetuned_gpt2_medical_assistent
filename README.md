# Fine-Tuned GPT-2 Medical Assistant

This repository contains a fine-tuned version of GPT-2 Large using PEFT (Parameter-Efficient Fine-Tuning) and QLoRA techniques for medical assistant tasks. The fine-tuning process was done to enhance the capabilities of GPT-2 in understanding and generating medical-related content.

## Model Overview
- **Base Model:** GPT-2 Large
- **Training Library:** PEFT and QLoRA
- **Pipeline Tag:** Text Generation
- **Purpose:** Medical Assistant/Consultation Chatbot

## Training Process

- **Hardware:** NVIDIA RTX 3050
- **Training Time:** Approximately 5 hours
- **Fine-Tuning Framework:** PEFT with QLoRA

The training process involved leveraging PEFT and QLoRA to achieve efficient model performance while significantly reducing the computational overhead. The RTX 3050 facilitated the training process, and we achieved this level of optimization within 5 hours.

## Features
- Improved contextual understanding of medical inquiries.
- Efficient generation of relevant and meaningful medical responses.
- Enhanced architecture to align with fine-tuning objectives.

## Usage Guide

Clone the repo and install the necessary dependencies by running:

```bash
git clone https://github.com/plk456/finetuned_gpt2_medical_assistent.git
cd finetuned_gpt2_medical_assistent
pip install -r requirements.txt
```

You can use the fine-tuned model for medical text generation tasks by running:

```python
from transformers import pipeline

# Load the model
generator = pipeline('text-generation', model='path_to_your_fine_tuned_model')

# Generate text
prompt = "What are the symptoms of diabetes?"
response = generator(prompt, max_length=50)
print(response)
```

## Repository Contents
- **`model/`**: Contains the fine-tuned model files.
- **`data/`**: Data used for fine-tuning the model.

## Acknowledgments
This project wouldn't have been possible without the open-source libraries and tools developed by the ML community, including Hugging Face libraries and the PEFT/QLoRA framework.

## Disclaimer
This model is intended for educational and research purposes only. It is not a replacement for professional medical advice, diagnosis, or treatment.

## Dataset Repo
https://github.com/abachaa/MedQuAD 
This is the dataset i used in this repo.
