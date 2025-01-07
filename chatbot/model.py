from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
import torch
def train_model(tokenizer, training_data, eval_data , model_name="t5-small", output_dir="./fin_intel_model"): # Add the eval_data argument
    """Fine-tunes a Gemini model on provided training data."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=1, # Reduced batch size
        learning_rate=2e-5,
        evaluation_strategy="no",  # Keep evaluation per epoch
        save_strategy="epoch", 
        #load_best_model_at_end=True # Removed loading of best model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
         eval_dataset=eval_data, # Add this argument to Trainer
    )

    trainer.train() 
    trainer.save_model(output_dir)
    
    return model