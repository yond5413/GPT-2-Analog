from datasets import load_dataset
from evaluate import load
from collections import OrderedDict, defaultdict
import numpy as np
import wandb
from transformers import AutoTokenizer
########
'''
File manages datasets for benchmarks
queried from Hugging Face API
'''
########
MAX_LENGTH = 1000#320
DOC_STRIDE = 128
#constants
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")#GPT2Model.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})

categories =['A', 'B', 'C','D']
labels = {categories[i]:i for i in range(len(categories))}
def load_race():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    #race = load_dataset("EleutherAI/race") ####RACE### dataset
    race = load_dataset("ehovy/race",'middle') ## ->requires either (all,high,or middle as second param)
   # ehovy/race nicer format but larger
    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    tokenized_data = race.map(
        preprocess_train, batched=True, remove_columns=race["train"].column_names
    )
    eval_data = race["validation"].map(
        preprocess_validation, batched=True, remove_columns=race["validation"].column_names
    )

    return race, tokenized_data, eval_data
###########################################################################
def preprocess_train(dataset):
    """Preprocess the training dataset"""
    prompt1 = f"Article: {dataset['article']}\n"
    prompt2 = f"Question: {dataset['question']}\n"
    prompt3 = f"Options: A){dataset['article'][0]},B){dataset['article'][1]}, C){dataset['article'][2]}, D){dataset['article'][3]}"
    full_prompt = prompt1 + prompt2 + prompt3
    #print(dataset['answer'])
    print("dataset length")
    print(len(dataset))
    print(f"{dataset.num_rows}")
   
    label =  [labels[i] for i in dataset['answer']]#[labels[i for i in dataset['answer']]]#choice[dataset['choice']]
    tokenized_dataset = TOKENIZER(full_prompt,padding="max_length", stride=DOC_STRIDE,max_length=MAX_LENGTH,truncation=True)
    tokenized_dataset['label'] = label 
    #-> add tokenizer
    for key, value in tokenized_dataset.items():
        print(f"Length of {key}: {len(value)}")
    print(len(tokenized_dataset))
    return tokenized_dataset

def preprocess_validation(dataset):
    """Preprocess the validation set"""
    prompt1 = f"Article: {dataset['article']}\n"
    prompt2 = f"Question: {dataset['question']}\n"
    prompt3 = f"Options: A){dataset['article'][0]},B){dataset['article'][1]}, C){dataset['article'][2]}, D){dataset['article'][3]}"
    full_prompt = prompt1 + prompt2 + prompt3
    label = labels[dataset['answer']]#label = choice[dataset['choice']]
    tokenized_dataset = TOKENIZER(full_prompt,padding="max_length", stride=DOC_STRIDE,max_length=MAX_LENGTH,truncation=True)
    tokenized_dataset['label'] = label 
    #-> add tokenizer
    return tokenized_dataset
##TODO

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


def race_inference(model, trainer, squad, eval_data, writer, max_inference_time=1e6, n_times=9,wandb_used = True):
    """Perform inference experiment at weight noise level specified at runtime.
    SQuAD exact match and f1 metrics are captured in Tensorboard
    """

    # Helper functions
    def predict():
        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(
            squad["validation"], eval_data, raw_predictions.predictions
        )

        # Format to list of dicts instead of a large dict
        #formatted_preds = [{"headline": k, "prediction": v} for k, v in predictions.items()]
        formatted_preds = predictions#-> convert to classes?
        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)

        return out_metric["f1"], out_metric["exact_match"]

    def write_metrics(f1, exact_match, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/f1", f1, t_inference)
        writer.add_scalar("val/exact_match", exact_match, t_inference)

        if wandb_used:#if ARGS.wandb:## TODO to param update
            wandb.log({"t_inference": t_inference, "f1": f1, "exact_match": exact_match})

        print(f"Exact match: {exact_match: .2f}\t" f"F1: {f1: .2f}\t" f"Drift: {t_inference: .2e}")

    model.eval()

    metric = load("squad")

    ground_truth = [row['choice'] for row in squad["validation"]]
    #ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad["validation"]]

    t_inference_list = np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()

    # Get the initial metrics
    f1, exact_match = predict()
    write_metrics(f1, exact_match, 0.0)

    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        f1, exact_match = predict()
        write_metrics(f1, exact_match, t_inference)

  