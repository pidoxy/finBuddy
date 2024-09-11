from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

def train_model(tokenizer, training_data, model_name="google/flan-t5-base", output_dir="./fin_intel_model"):
    """Fine-tunes a Gemini model on provided training data."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=8,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,  # Assuming you have processed this appropriately
        tokenizer=tokenizer,
    )

    trainer.train() 
    model.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    
    return model