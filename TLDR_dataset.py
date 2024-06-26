from datasets import load_dataset
from evaluate import load
from collections import OrderedDict, defaultdict
import numpy as np
import wandb
from transformers import AutoTokenizer, GPT2Tokenizer
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
########
'''
File manages datasets for benchmarks
queried from Hugging Face API
'''
########
MAX_LENGTH = 1000
DOC_STRIDE = 128
#constants
# -> source
categories =['Sponsor', 'Big Tech & Startups', 'Science and Futuristic Technology',
                           'Programming, Design & Data Science', 'Miscellaneous']
labels = {i:categories[i] for i in range(len(categories))}
#TOKENIZER = AutoTokenizer.from_pretrained("gpt2")#GPT2Model.from_pretrained(MODEL_NAME)
TOKENIZER =GPT2Tokenizer.from_pretrained('gpt2')
def load_tldr():
    """Load the TLDR dataset, the tokenized version, and the validation set"""
    tldr = load_dataset("JulesBelveze/tldr_news")
    ##############################
    tldr_val =tldr['test']
    tldr.pop('test')
    tldr['validation'] = tldr_val ## just for sake of naming convention in trainer function
    ##############################
    print('Preprocessing Training set')
    train_set= preprocess_dataset(tldr['train'])
    print('Preprocessing Validation set')
    val_set = preprocess_dataset(tldr['validation'])
    return tldr, train_set,val_set#tokenized_data, eval_data
###########################################################################
def preprocess_dataset(dataset):
    """Preprocess the training dataset"""
    ##updates based off tldr
    '''-> headline, content, category
    '''
    ## dataset["question"] = [q.lstrip() for q in dataset["question"]] 
    ret =[]
    progress_bar = tqdm(total=len(dataset))
    for i in range(len(dataset)):
        curr = {}
        prompt = f"headline: {dataset['headline'][i]} \n context:{dataset['headline'][i]} "
    #tokenized_dataset = TOKENIZER(prompt,padding="max_length", stride=DOC_STRIDE,max_length=MAX_LENGTH,truncation=True)
        category = dataset['category'][i]
        #print(f'category:{category}, label:{labels[category]}')
        curr['prompt'] = prompt
        curr['target']  = category#labels[category]
        ret.append(curr)#ret[i] = curr
        progress_bar.update(1)
    return ret 

def postprocess_predictions(pred):
    scores = np.array(pred)
    index = np.argmax(scores)
    return index#predictions

def tldr_inference(ARGS,model, tldr, eval_data, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime.
    TLDR exact match and f1 metrics are captured in Tensorboard
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Helper functions
    def exact_match(pred,gt):
        ret = 0.0
        for i in range(len(pred)):
            if pred[i] == gt[i]:
                ret+=1.0
        ret /= len(pred)
        return ret
    
    def predict():
        # Perform inference + evaluate metric here
        pred = []
        model.to(device)
        #print(f"device for sanity: {device}")
        progress_bar = tqdm(total=len(eval_data))
        for sample in eval_data:       
            #raw_predictions = trainer.predict(eval_data)
            scores= []
            #prompt_toks=TOKENIZER(sample['prompt'], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)[0]
            prompt_toks = TOKENIZER.encode(sample['prompt'], return_tensors="pt")[0]
            #print(prompt_toks)
            #print(len(prompt_toks))
            prompt_tok_count = prompt_toks.numel()
            #print('c ->loops')
            for c in categories: ##-> class labels
                curr = f'{sample} {c}'
                input_ids = TOKENIZER.encode(curr, return_tensors="pt")#, max_length=MAX_LENGTH, truncation=True)
                #input_ids = TOKENIZER(sample['prompt'], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
                toks_pred = input_ids[0].numel() - prompt_tok_count
                
                
                input_ids = input_ids.to(device)
                # Synchronize
                torch.cuda.synchronize()
                #print(toks_pred)
                #print(input_ids)
                #print("in_ids device:", input_ids.device)
                #with torch.no_grad():
                outputs = model(*input_ids)
                logits = outputs.logits
                probs = torch.softmax(logits,dim=-1)
                targ_probs = probs[-toks_pred:]
                argmaxs = torch.argmax(targ_probs,dim=-1)
                maxes = targ_probs[torch.arange(targ_probs.size(0)), argmaxs] #max[-toks_pred:] 
                scores.append(torch.max(maxes).item())
            pred_index = postprocess_predictions(scores)
            pred.append(pred_index)    
                #print(logits.size())
                #predicted_index = torch.max#torch.argmax(outputs.logits)
                #pred.append(predicted_index.item())
            progress_bar.update(1)
            
        formatted_preds = pred
        
        micro_f1 = f1_score(ground_truth, formatted_preds, average='micro')

        # Compute macro F1 score
        macro_f1 = f1_score(ground_truth, formatted_preds, average='macro')

        # Compute weighted F1 score
        weighted_f1 = f1_score(ground_truth, formatted_preds, average='weighted')

        em = exact_match(formatted_preds,ground_truth)

        return micro_f1,macro_f1,weighted_f1,em

    def write_metrics(micro_f1,macro_f1,weighted_f1,em, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/micro_f1", micro_f1, t_inference)
        writer.add_scalar("val/macro_f1", macro_f1, t_inference)
        writer.add_scalar("val/weighted_f1", weighted_f1, t_inference)
        writer.add_scalar("val/exact_match", em, t_inference)
        if ARGS.wandb:
            wandb.log({"t_inference": t_inference, "micro_f1": micro_f1, "macro_f1":macro_f1,
                       "weighted_f1":weighted_f1,"exact_match": em})

        print(f"Exact match: {em: .2f}\t" f"Micro F1: {micro_f1: .2f}\t" f"Drift: {t_inference: .2e}")
        print(f"Macro F1: {macro_f1: .2f}\t" f"Weighted F1: {weighted_f1: .2f}\t")
    model.eval()

    ground_truth = [row['category'] for row in tldr["validation"]]
    
    t_inference_list = np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()

    # Get the initial metrics
    micro_f1,macro_f1,weighted_f1,em = predict()
    write_metrics(micro_f1,macro_f1,weighted_f1,em,0.0)
    if not ARGS.digital:
        for t_inference in t_inference_list:
            model.drift_analog_weights(t_inference)
            micro_f1,macro_f1,weighted_f1,em = predict()
            write_metrics( micro_f1,macro_f1,weighted_f1,em,t_inference)
    else:
        print("Not analog just GPT-2 baseline")

