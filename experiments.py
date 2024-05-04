import os
from argparse import ArgumentParser
from SQuAD_dataset import get_SQuAD
#MODEL_NAME = "gpt2"#openai-community/gpt2
MODEL_NAME = "gpt2openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
based_model = GPT2Model.from_pretrained(MODEL_NAME)

def main():
    pass
if __name__ == "__main__":
    ### -> dataset file for each of 
    ### the files and do preprocessing/inference from there
    ### another file to manage aihwkit configs too?
    PARSER = ArgumentParser("Analog GPT-2")
    ## will add rest from notebook
    
    
    main()