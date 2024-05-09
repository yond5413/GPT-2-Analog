import os
from datetime import datetime
from argparse import ArgumentParser
from TLDR_dataset  import load_tldr, tldr_inference, postprocess_predictions
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
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)#GPT2Model.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
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
    "parameters": {"modifier_noise": {"values": [0, 0.05, 0.1, 0.2]}},
                  "digital":{"values":[True,False]},
                    "load":{"values":[False,True]},
                    "ideal":{"values":[False,True]} 
    }
    
    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project='GPT2-analog')#"gpt2-weight-noise-experiment")
#############################
def train(model,train,optimizer,epochs = 3):
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
            scores= []
            prompt_toks=tokenizer.encode(sample['prompt'], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)[0]
            prompt_tok_count = prompt_toks.numel()
            for c in categories:
                text = sample['prompt']+' ' + c
                input_ids = tokenizer.encode(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
                #input_ids = TOKENIZER(sample['prompt'], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
                toks_pred = input_ids[0].numel() - prompt_tok_count
                input_ids = input_ids.to(device)
                # Synchronize
                torch.cuda.synchronize()
               
                outputs = model(*input_ids)
                logits = outputs.logits
                probs = torch.softmax(logits,dim=-1)
                targ_probs = probs[-toks_pred:]
                argmaxs = torch.argmax(targ_probs,dim=-1)
                maxes = targ_probs[torch.arange(targ_probs.size(0)), argmaxs] #max[-toks_pred:] 
                scores.append(torch.max(maxes).item())
            pred_index = postprocess_predictions(scores)
            pred  =torch.tensor(pred_index,dtype=torch.float32).to(device)#predicted_index.to(torch.float32)
            gt = torch.tensor(sample['target'],dtype=torch.float32).to(device)
            loss = F.mse_loss(pred,gt)#F.cross_entropy(pred, gt)
            gt.requires_grad_(True)
            pred.requires_grad_(True)
            total_loss += loss.item()
            loss.requires_grad_(True)
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            progress_bar.update(1)
        print(f"Epoch {i}, Average Loss: {total_loss / len(train)}")
        
def make_writer():
    f'with_noise_{args.noise}' 
    log_dir = "logs/fit/" + args.run_name+f'with_noise_{args.noise}_ideal:{args.ideal}' 
    writer = SummaryWriter(log_dir=log_dir)
    wandb.tensorboard.patch(root_logdir=log_dir)
    return writer

def main():
    if wandb:
        wandb.init()
        #print(wandb.config)
        #print(wandb.config.modifier_noise)
        args.digital = wandb.config.digital
        args.load = wandb.config.load
        args.ideal = wandb.config.ideal
        args.noise = wandb.config.modifier_noise
        #print(f"digital: {args.digital}")
        #print(f"loading:{args.load}")
    #num_classes = 5
    init_dataset, train_set, val_set = load_tldr()
    
    model = get_model(args)#,num_classes)
    optimizer = create_optimizer(model,args.learning_rate)
   
    writer = make_writer()
   
    if args.load:
        print(f"Loading model from '{args.checkpoint}'.")
        model.load_state_dict(torch_load(args.checkpoint))

    # Do hw-aware training if in analog domain and the model isn't loaded from
    # an existing checkpoint
    if args.train_hwa and not args.digital and not args.load:
        #print("Hardware aware training.......")
        #### not implemented.......
        #trainer.train()
        pass
        #train(model,train_set,optimizer)
        #torch_save(model.state_dict(), args.checkpoint)
    if args.digital: #and not args.load:
        print("default gpt-2 with finetuning")
        train(model,train_set,optimizer)
        torch_save(model.state_dict(), args.checkpoint)
    print("TLDF dataset inference......")
    #tldr_inference(args,model, trainer, init_dataset, eval_data, writer)
    tldr_inference(args,model,init_dataset, val_set, writer)
if __name__ == "__main__":
    if args.wandb:
        #wandb.init()
        wandb.agent(SWEEP_ID, function=main, count=5)
    else:
        main()