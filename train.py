from datasets import JsonDataset
from config import paths, model, data_collator, training_args, model_options
from callbacks import CustomDefaultFlowCallback, DefaultFlowCallback
from transformers import Trainer


# Create the train and evaluation dataset
train_dataset = JsonDataset(paths["train_path"])
eval_dataset = JsonDataset(paths["dev_path"])

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # prediction_loss_only=True,
)

if model_options["output_from_model"]:
    trainer.remove_callback(DefaultFlowCallback)
    trainer.add_callback(CustomDefaultFlowCallback)

# Train the model
trainer.train(resume_from_checkpoint=model_options["resume-from-checkpoint"])
