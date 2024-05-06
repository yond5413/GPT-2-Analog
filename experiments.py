import os
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
from model_config import get_model,create_optimizer
#MODEL_NAME = "gpt2"#openai-community/gpt2
MODEL_NAME = "gpt2"
#tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
#based_model = GPT2Model.from_pretrained(MODEL_NAME)
#model =AutoModelForCausalLM.from_pretrained("gpt2")# GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained("gpt2")#GPT2Model.from_pretrained(MODEL_NAME)

####
#wanddb?
###
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
    PARSER = ArgumentParser("Analog GPT-2")
    ## TODO add the args 
    model = get_model(args.ideal,args.noise)
    optimizer = create_optimizer(model,args.learning_rate)
    ### arg.dataset-->> will add 
    if True:
        init_dataset, tokenized_data, eval_data = load_squad()
    elif True:
        num_classes = 4
        init_dataset, tokenized_data, eval_data = load_race()
    elif True:
        num_classes = 5
        init_dataset, tokenized_data, eval_data = load_tldr()
    trainer = make_trainer(model=model,optimizer=optimizer,tokenized_data=tokenized_data)