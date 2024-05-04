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
def download_dataset(input:int):
  '''
  Input -> integer will decide which dataset to load from Hugging Face
  only accept values between 0-4 from datasets
  TLDR_NEWs, SQUAD, GLUE, RACE, and LAMBDA
  '''
  if input == 0:
    dataset = load_dataset("JulesBelveze/tldr_news")
  elif input ==1:
    dataset = load_dataset("rajpurkar/squad")
  elif input ==2:
    dataset = load_dataset("nyu-mll/glue")
  elif input ==3:
   dataset = load_dataset("1704.04683") ####RACE###
  elif input ==4:
    displayataset = load_dataset("cimec/lambada")
  else:
    dataset = None
    print("Invalid")
  return dataset

def create_datasets():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    squad = download_dataset(ARGS.data_set)#load_dataset("squad")
    if squad ==None:
        return None
    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    tokenized_data = squad.map(
        preprocess_train, batched=True, remove_columns=squad["train"].column_names
    )
    eval_data = squad["validation"].map(
        preprocess_validation, batched=True, remove_columns=squad["validation"].column_names
    )

    return squad, tokenized_data, eval_data
###########################################################################
def preprocess_train(dataset):
    """Preprocess the training dataset"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and padding,
    # but keep the overflows using a stride. This results
    # in one example possibly giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature, the stride being the number
    # of overlapping tokens in the overlap.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to
    # character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_dataset.pop("offset_mapping")

    # Store start and end character positions for answers in context
    tokenized_dataset["start_positions"] = []
    tokenized_dataset["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_dataset["input_ids"][i]
        cls_index = input_ids.index(TOKENIZER.cls_token_id)

        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)

        # One example can give several spans, this
        # is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = dataset["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_dataset["start_positions"].append(cls_index)
            tokenized_dataset["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span
            # (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and
                # token_end_index to the two ends of the answer.
                # Note: we could go after the last offset
                # if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_dataset["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_dataset["end_positions"].append(token_end_index + 1)

    return tokenized_dataset


def preprocess_validation(dataset):
    """Preprocess the validation set"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and maybe padding,
    # but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1

        # One example can give several spans,
        # this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])

        # Set to None the offset_mapping that are not
        # part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_dataset["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
        ]

    return tokenized_dataset
##TODO
'''
preprocessing updates 
-> 
'''
if __name__ == "__main__":
  print("test")

  print("success")