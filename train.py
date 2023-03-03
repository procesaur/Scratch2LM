from datasets import JsonDataset
from config import train_path, dev_path, model, tokenizer, device, data_collator, training_args
from callbacks import CustomDefaultFlowCallback, DefaultFlowCallback
from transformers import Trainer


# Create the train and evaluation dataset
train_dataset = JsonDataset(train_path)
eval_dataset = JsonDataset(dev_path)

# Initialize the model from a configuration without pretrained weights
print(device)
model = model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # prediction_loss_only=True,
)

trainer.remove_callback(DefaultFlowCallback)
trainer.add_callback(CustomDefaultFlowCallback)

# Train the model
trainer.train()
