from test import test
from transformers import DefaultFlowCallback, ProgressCallback
from transformers.trainer_callback import TrainerState, TrainerControl, TrainingArguments, IntervalStrategy
from config import model_folder, tokenizer_path, tokenizer


class CustomDefaultFlowCallback(DefaultFlowCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % args.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True
            examples = test(kwargs["model"])
            print(examples)

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control