import os
from datetime import datetime
from argparse import ArgumentParser
from TLDR_dataset  import load_tldr, tldr_inference
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
from tqdm import tqdm
import torch
import torch.nn.functional as F
#MODEL_NAME = "gpt2"#openai-community/gpt2
MODEL_NAME = "gpt2"
#tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
#based_model = GPT2Model.from_pretrained(MODEL_NAME)
#model =AutoModelForCausalLM.from_pretrained("gpt2")# GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained("gpt2")#GPT2Model.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#############################
# cli parser
MAX_LENGTH = 1000#320
DOC_STRIDE = 128
PARSER = ArgumentParser("Analog GPT-2")
PARSER.add_argument("-d", "--digital", help="Add to use digital inference", type = bool, default= True)
PARSER.add_argument(
    "-i",
    "--ideal",
    help="Add to use ideal config instead of default noisy one",
    type = bool, default= True
    #action="store_true",
)
PARSER.add_argument("-w", "--wandb", help="Add to use wandb", type= bool, default=True)
PARSER.add_argument("-n", "--noise", help="Modifier noise", default=0.1, type=float)
PARSER.add_argument(
    "-r",
    "--run_name",
    help="Tensorboard run name",
    default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    type=str,
)
PARSER.add_argument("-t", "--train_hwa", help="Use Hardware-Aware training",type = bool, default= True)# action="store_true")
PARSER.add_argument(
    "-L", "--load", help="Use when loading from training checkpoint", type = bool, default= False#True#action="store_true"
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
'''PARSER.add_argument(
    "-ds","--dataset",help="dataset flag (0/1) for TLDR or RACE",default=0,type=int
)'''
args = PARSER.parse_args()
#############################
### seting up wandb ie weights and biases 
if args.wandb:
    # Define weights noise sweep configuration
    SWEEP_CONFIG = {
        "method": "random",
        "name": "modifier noise sweep",
        "metric": {"goal": "maximize", "name": "exact_match"},
    "parameters": {"modifier_noise": {"values": [0,0, 0.05, 0.1, 0.2]},
                  "digital":{"values":[True,False,False,False,False]},
                    "load":{"values":[False,True,True,True,True]} },
    }

    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project='GPT2-analog')#"gpt2-weight-noise-experiment")
#############################
def train(model,train,optimizer,epochs = 25):
    categories =['Sponsor', 'Big Tech & Startups', 'Science and Futuristic Technology',
                           'Programming, Design & Data Science', 'Miscellaneous']
    labels = {i:categories[i] for i in range(len(categories))}
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0,epochs):
        print(f"Epoch: {i}")
        total_loss = 0.0
        progress_bar = tqdm(total=len(train))
        for sample in train:       
            #raw_predictions = trainer.predict(eval_data)
            model.zero_grad()
            '''
            for c in categories
                text = sample['prompt']+' ' + c
            '''
            input_ids = tokenizer(sample['prompt'], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
            input_ids.to(device)
            ###################
            outputs = model(**input_ids)
            predicted_index = torch.argmax(outputs.logits)
            pred = predicted_index.to(torch.float32)
            gt = torch.tensor(sample['target'],dtype=torch.float32).to(device)
            
            gt.requires_grad_(True)
            pred.requires_grad_(True)
            #print(pred)
            #print(gt)
            loss = F.cross_entropy(pred, gt)#F.mse_loss(pred,gt)#F.cross_entropy(pred, gt)
            total_loss += loss.item()
            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update progress bar and total loss
            
            progress_bar.update(1)
        print(f"Epoch {i}, Average Loss: {total_loss / len(train)}")
        
def make_writer():
    log_dir = "logs/fit/" + args.run_name
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def main():
    if wandb:
        print('calma')
        print(wandb.config.modifier_noise)
        args.digital = wandb.config.digital
        args.load = wandb.config.load
        print(f"digital: {args.digital}")
        print(f"loading:{args.load}")
    num_classes = 5
    init_dataset, train_set, val_set = load_tldr()
    
    model = get_model(args,num_classes)
    optimizer = create_optimizer(model,args.learning_rate)
    #trainer,writer = make_trainer(model=model,optimizer=optimizer,tokenized_data=tokenized_data)
    writer = make_writer()
    '''
    ->Change args to correct 
    --> setup for correct inference function for the dataset being used
    '''
    if args.load:
        print(f"Load model from '{args.checkpoint}'.")
        model.load_state_dict(torch_load(args.checkpoint))

    # Do hw-aware training if in analog domain and the model isn't loaded from
    # an existing checkpoint
    if args.train_hwa and not args.digital and not args.load:
        print("Hardware aware training.......")
        #trainer.train()
        #train(model,train_set,optimizer)
        torch_save(model.state_dict(), args.checkpoint)
    if args.digital: #and not args.load:
        print("default gpt-2 with finetuning")
        train(model,train_set,optimizer)
        torch_save(model.state_dict(), args.checkpoint)
    print("TLDF dataset inference......")
    #tldr_inference(args,model, trainer, init_dataset, eval_data, writer)
    tldr_inference(args,model,init_dataset, val_set, writer)
if __name__ == "__main__":
    if args.wandb:
        wandb.init()
        wandb.agent(SWEEP_ID, function=main, count=5)
    else:
        main()