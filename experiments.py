import os
from datetime import datetime
from argparse import ArgumentParser
from SQuAD_dataset import load_squad
from RACE_dataset  import load_race
from TLDR_dataset  import load_tldr
###########################################################
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering, ## not needed but was there in bert example
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)
from transformers import GPT2Tokenizer, GPT2Model
###########################################################
from torch import save as torch_save, load as torch_load
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
###########################################################
import wandb
from model_config import get_model,create_optimizer
#MODEL_NAME = "gpt2"#openai-community/gpt2
MODEL_NAME = "gpt2"
#tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
#based_model = GPT2Model.from_pretrained(MODEL_NAME)
#model =AutoModelForCausalLM.from_pretrained("gpt2")# GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained("gpt2")#GPT2Model.from_pretrained(MODEL_NAME)
#############################
# cli parser
PARSER = ArgumentParser("Analog GPT-2")
PARSER.add_argument("-d", "--digital", help="Add to use digital inference", action="store_true")
PARSER.add_argument(
    "-i",
    "--ideal",
    help="Add to use ideal config instead of default noisy one",
    action="store_true",
)
PARSER.add_argument("-w", "--wandb", help="Add to use wandb", action="store_true")
PARSER.add_argument("-n", "--noise", help="Modifier noise", default=0.1, type=float)
PARSER.add_argument(
    "-r",
    "--run_name",
    help="Tensorboard run name",
    default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    type=str,
)
PARSER.add_argument("-t", "--train_hwa", help="Use Hardware-Aware training", action="store_true")
PARSER.add_argument(
    "-L", "--load", help="Use when loadiung from training checkpoint", action="store_true"
)

PARSER.add_argument(
    "-c",
    "--checkpoint",
    help="File name specifying where to load/save a checkpoint",
    default="./saved_chkpt.pth",
    type=str,
)
PARSER.add_argument(
    "-l", "--learning_rate", help="Learning rate for training", default=2e-4, type=float
)
PARSER.add_argument(
    "-ds","--dataset",help="dataset flag (0/1) for TLDR or RACE",default=0,type=int
)
args = PARSER.parse_args()
#############################
### seting up wandb ie weights and biases 
if args.wandb:
   

    # Define weights noise sweep configuration
    SWEEP_CONFIG = {
        "method": "random",
        "name": "modifier noise sweep",
        "metric": {"goal": "maximize", "name": "exact_match"},
        "parameters": {"modifier_noise": {"values": [0, 0.05, 0.1, 0.2]}},
    }

    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project="gpt2-weight-noise-experiment")
#############################

def make_trainer(model, optimizer, tokenized_data):
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir="./", ###-> add dataset arg
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.001,
        no_cuda=False,
    )

    collator = DefaultDataCollator()

    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)],
    )

    return trainer, writer

def main():
    pass
if __name__ == "__main__":
    ### -> dataset file for each of 
    ### the files and do preprocessing/inference from there
    ### another file to manage aihwkit configs too?
    ## TODO add the args 
    model = get_model(args.ideal,args.noise)
    optimizer = create_optimizer(model,args.learning_rate)
    ### arg.dataset-->> will add 
    if args.dataset ==0:
        num_classes = 4
        init_dataset, tokenized_data, eval_data = load_race()
    elif args.dataset ==1:
        num_classes = 5
        init_dataset, tokenized_data, eval_data = load_tldr()
    else:
        ## error handling for invalid dataset 
        print("Invalid dataset value")
        print("Currently supports 0 or 1 for TLDR/RACE respectively")
        exit()
    trainer = make_trainer(model=model,optimizer=optimizer,tokenized_data=tokenized_data)