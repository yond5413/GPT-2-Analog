Welcome to my exploratory analysis of the GPT-2 model using the 
IBM AIHWkit 
### Dependencies ####
To install of the dependencies run this command in your cli

<code>pip install -r requirements.txt</code>

To utilize cuda with the aihwkit please install these as well!!

<code>wget https://aihwkit-gpu-demo.s3.us-east.cloud-object-storage.appdomain.cloud/aihwkit-0.9.0+cuda117-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
</code>

<code>pip install aihwkit-0.9.0+cuda117-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl</code>

### Sample commands to run program ###

<code>python experiments.py </code>

### Summary ###
The goal of this project is to research how the gpt-2 model will perform
across various datasets. The two most notable are SQUAD and RACE for LLM
benchmarks. I also choose TLDR_News as the dataset seemed like a good fit for
the task. Essentially the goal was to finetune gpt-2 on these datasets that 
are centered around NLU(Natural Language Understanding). Due to technical issues I 
had working on this I ended up only using the TLDR dataset, but I believe the results are still note worthy.

### Results: ###
Herre we can see the layers of the GPT-2 prior to converting and also after.
![base-gpt](results/gpt-2_defaults.png)
![analog-gpt](results/gpt-2_analog_wrapping.png)

Alongside this here are some of my best result from the evaluation metrics after finetuning for 3 epochs
I believe adjusting the optimizer when running in the digital mode will fix the non-ideal RPU conifg.

![ideal](results/ideal_results.png)
Here we have the non-ideal results with different noise values
![analog-gpt](results/non_ideal_results.png)
