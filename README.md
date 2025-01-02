# Fine-tuning BERT for Medical Reasoning

This repository contains the implementation of a fine-tuned BERT-based model on the [Hugging Face Medical-O1 Reasoning Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT). The model is designed to perform structured medical reasoning and generate contextually relevant responses to medical queries. 

## Overview

Medical reasoning is a critical task in clinical decision-making, and this project aims to simulate this process using a fine-tuned transformer-based language model. The model handles structured inputs comprising questions, reasoning steps, and responses, enabling it to provide accurate and insightful answers.

Key features:
- Fine-tuning a `bert-base-cased` model for medical reasoning tasks.
- Preprocessing and formatting data for causal language modeling.
- Training pipeline using Hugging Face Transformers and PyTorch.
- Evaluation and inference for real-world medical queries.

## Dataset

The project uses the [Medical-O1 Reasoning Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT), which includes structured inputs with fields like:
- `Question`: Medical question posed by a user.
- `Complex_CoT`: Chain-of-thought reasoning supporting the response.
- `Response`: Final medical advice or answer.

## Dependencies

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Datasets
- NumPy
- Matplotlib (optional, for visualization)

Install dependencies using:
```bash
pip install torch transformers datasets numpy
```

## How It Works

1. **Dataset Loading**: The dataset is loaded using the Hugging Face `datasets` library and split into training and validation subsets.
2. **Data Preprocessing**: Input data is tokenized and formatted to include the question, reasoning, and response for causal language modeling.
3. **Model Fine-tuning**: The `bert-base-cased` model is fine-tuned using a training loop with customized hyperparameters.
4. **Evaluation and Inference**: The trained model is tested and used to generate responses to unseen medical queries.


## Training

To train the model, run:
```bash
python main.py
```

Hyperparameters:
- Learning Rate: `2e-5`
- Batch Size: `4`
- Epochs: `3`
- Gradient Accumulation: `4`

## Inference

To generate a response for a medical query:

### Using Custom Inference Code
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./medical_reasoning_final_model")
model = AutoModelForCausalLM.from_pretrained("./medical_reasoning_final_model")

# Function to generate response
def generate_medical_response(question, model, tokenizer, max_length=512):
    prompt = f"Question: {question}\nThinking:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        model = model.to("cuda")

    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
question = "I have been experiencing severe headaches and dizziness for the past week. What could be wrong?"
response = generate_medical_response(question, model, tokenizer)
print(response)
```

### Using Hugging Face Pipeline
Alternatively, you can use the Hugging Face `pipeline` for simplified inference:

```python
from transformers import pipeline

# Load the fine-tuned model using pipeline
medical_pipeline = pipeline("text-generation", model="./medical_reasoning_final_model")

# Generate a response for a medical query
question = "I have been experiencing severe headaches and dizziness for the past week. What could be wrong?"
prompt = f"Question: {question}\nThinking:"
response = medical_pipeline(prompt, max_length=512, num_return_sequences=1, temperature=0.7, top_p=0.9)

print(response[0]["generated_text"])
```

## Results

The fine-tuned model successfully replicates structured medical reasoning and generates coherent responses. It is capable of:
- Understanding complex medical queries.
- Simulating reasoning steps to derive a response.
- Providing clinically relevant and context-aware answers.

## Future Work

- Experiment with larger models like LLaMA-2 for improved performance.
- Incorporate domain-specific reinforcement learning for better alignment with clinical practices.
- Extend the dataset to include diverse medical scenarios.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments

- [Hugging Face](https://huggingface.co) for the Transformers library and datasets.
- [Freedom Intelligence](https://huggingface.co/FreedomIntelligence) for providing the Medical-O1 Reasoning Dataset.

## Contact

For questions or feedback, please contact [Rishabh Dewangan](mailto:rishabhcicdu@gmail.com).
