from datasets import load_dataset
from evaluate import load
########
'''
File manages datasets for benchmarks
queried from Hugging Face API
'''
########
MAX_LENGTH = 320
DOC_STRIDE = 128
#constants
  
def create_datasets():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    race = load_dataset("1704.04683") ####RACE### dataset
   
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

    return tokenized_dataset


def preprocess_validation(dataset):
    """Preprocess the validation set"""
    
    return tokenized_dataset
##TODO
'''
preprocessing updates 
-> 
'''
if __name__ == "__main__":
  print("test")

  print("success")