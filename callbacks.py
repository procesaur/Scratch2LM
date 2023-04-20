from test import test
from transformers import DefaultFlowCallback
from transformers.trainer_callback import TrainerState, TrainerControl, TrainingArguments, IntervalStrategy
from config import paths, model_options
from torch import save


class CustomDefaultFlowCallback(DefaultFlowCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        # Save model?
        if model_options["save_each_epoch"]:
            save(kwargs["model"], paths["model_folder"] + "/epoch_" + str(state.epoch))

        return control

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

            # Perform Experiment?
            if model_options["output_from_model"]:
                examples = test(kwargs["model"])
                if not isinstance(examples[0], str):
                    examples = [e for ee in examples for e in ee]
                with open(paths["model_folder"] + "/experiments.log", "a+", encoding="utf-8") as lf:
                    lf.write("\t".join(examples))
                    lf.write("\n")

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control
