from datasets import load_dataset
from evaluate import load
from collections import OrderedDict, defaultdict
import numpy as np
import wandb
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
########
'''
File manages datasets for benchmarks
queried from Hugging Face API
'''
########
MAX_LENGTH = 320
DOC_STRIDE = 128
#constants
# -> source
categories =['Sponsor', 'Big Tech & Startups', 'Science and Futuristic Technology',
                           'Programming, Design & Data Science', 'Miscellaneous']
labels = {i:categories[i] for i in range(len(categories))}
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")#GPT2Model.from_pretrained(MODEL_NAME)
def load_tldr():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    tldr = load_dataset("JulesBelveze/tldr_news")
    ##############################
    tldr_val =tldr['test']
    tldr.pop('test')
    tldr['validation'] = tldr_val ## just for sake of naming convention in trainer function
    ##############################
    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    #tokenized_data = tldr.map(
     #   preprocess_train, batched=True, remove_columns=tldr["train"].column_names
    #)
    #eval_data = tldr["validation"].map(
    #    preprocess_validation, batched=True, remove_columns=tldr["validation"].column_names
    #)
    print('Preprocessing Training set')
    train_set= preprocess_train(tldr['train'])
    print('Preprocessing Validation set')
    val_set = preprocess_train(tldr['validation'])
    return tldr, train_set,val_set#tokenized_data, eval_data
###########################################################################
def preprocess_train(dataset):
    """Preprocess the training dataset"""
    ##updates based off tldr
    '''-> headline, content, category
    '''
    ## dataset["question"] = [q.lstrip() for q in dataset["question"]] 
    ## ex for preprocessing---->
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

def postprocess_predictions(examples, features, raw_predictions,):
    #features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    print(
        f"Post-processing {len(examples)} example predictions "
        f"split into {len(features)} features."
    )
    predictions = []#OrderedDict()
    predicted_class = []
    for val, logits in (features,raw_predictions):
        # Take the argmax of the logits to get the predicted class index
        predicted_class_idx = np.argmax(logits)
        # Map the class index to the corresponding label
        predicted_label = categories[predicted_class_idx]
        predictions.append(predicted_label)
        #predicted_class.append(predicted_class_idx)
    return predictions


def tldr_inference(ARGS,model, squad, eval_data, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime.
    SQuAD exact match and f1 metrics are captured in Tensorboard
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Helper functions
    def predict():
        # Perform inference + evaluate metric here
        pred = []
       
        progress_bar = tqdm(total=len(eval_data))
        for sample in eval_data:       
            #raw_predictions = trainer.predict(eval_data)
            input_ids = TOKENIZER(sample['prompt'], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
            input_ids.to(device)
            with torch.no_grad():
                outputs = model(**input_ids)
            predicted_index = torch.argmax(outputs.logits)
            pred.append(predicted_index)
            progress_bar.update(1)
            #predictions = postprocess_predictions(
            #    squad["validation"], eval_data, raw_predictions.predictions
            #)

            # Format to list of dicts instead of a large dict
            #formatted_preds = [{"headline": k, "prediction": v} for k, v in predictions.items()]
        formatted_preds = pred
        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)

        return out_metric["f1"], out_metric["exact_match"]

    def write_metrics(f1, exact_match, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/f1", f1, t_inference)
        writer.add_scalar("val/exact_match", exact_match, t_inference)

        if ARGS.wandb:
            wandb.log({"t_inference": t_inference, "f1": f1, "exact_match": exact_match})

        print(f"Exact match: {exact_match: .2f}\t" f"F1: {f1: .2f}\t" f"Drift: {t_inference: .2e}")

    model.eval()

    metric = load("squad")

    ground_truth = [row['category'] for row in squad["validation"]]
    #ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad["validation"]]

    t_inference_list = np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()

    # Get the initial metrics
    f1, exact_match = predict()
    write_metrics(f1, exact_match, 0.0)

    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        f1, exact_match = predict()
        write_metrics(f1, exact_match, t_inference)

