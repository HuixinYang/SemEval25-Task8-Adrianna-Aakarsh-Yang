# Monte-Carlo Tree Search for Large Language Models

This repository is a fork of [mcts-for-llm](https://github.com/shunzh/mcts-for-llm) which is a decode with Monte-Carlo tree search. It is capable to run one data point now for our code generation task(only manage to run without error, but very simply and roughly). But at least the structure works. 


Run the code-generator file in examples, .py file for local, notebook for colab.

### [01.26.2025]

- Generated Predictions Files: 

    - MISSING_SUBMISSION - 54 entries
    - CODE_ERROR - 2 entries
    - TRIVIAL CODE - 14 

Thus maximum accuracy achievable accuracy: 86.5 


### [01.11.2025]

Computed finetune accuracy by type: 

https://huggingface.co/datasets/aakarsh-nair/semeval-2025-task-8-finetune/sql-console/mMQoACm

boolean 480 271 0.564583
list[number] 480 15 0.03125
number 510 452 0.886275
category 570 381 0.668421
list[category] 540 55 0.101852

### [Yang.16.12.2024] 

This is what I have done so far. I feel the algorithms is very clear to me. But the hardest thing is the implementation, especially to figure out code details of mct_process in mct.py, also the related details. It took me some time.  

P.S. I will not continue to work on this task till 27 Dec, especially during 20 Dec to 27 Dec I am not here and will not work. 