from datasets import JsonDataset
from config import train_path, dev_path, model, data_collator, training_args, output_from_model, resume
from callbacks import CustomDefaultFlowCallback, DefaultFlowCallback
from transformers import Trainer


# Create the train and evaluation dataset
train_dataset = JsonDataset(train_path)
eval_dataset = JsonDataset(dev_path)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # prediction_loss_only=True,
)

if output_from_model:
    trainer.remove_callback(DefaultFlowCallback)
    trainer.add_callback(CustomDefaultFlowCallback)

# Train the model
trainer.train(resume_from_checkpoint=resume)
