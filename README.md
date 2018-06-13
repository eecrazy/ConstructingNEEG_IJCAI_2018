# ConstructingNEEG_IJCAI_2018

## Paper Data and Code
The data and code for our IJCAI-ECAI 2018 Paper: [Constructing Narrative Event Evolutionary Graph for Script Event Prediction](https://arxiv.org/abs/1805.05081).

The codes here include PyTorch implementations of the PairLSTM baseline and our SGNN model. Code for EventComp can be found [here](http://mark.granroth-wilding.co.uk/papers/what_happens_next/).

Data used in our paper can be found [here](https://drive.google.com/open?id=1WFBDL_zfNC1sSuz0dmaMux3w-OB_hUui).


## How to run the code?

You need to download the data I used from [google-drive](https://drive.google.com/open?id=1WFBDL_zfNC1sSuz0dmaMux3w-OB_hUui). Besides, you need Python3.5 or 3.6, PyTorch 0.3.0, and Nvidia GPU, perhaps Titan XP or Tesla P100. You can run python3 evaluate.py to get the results reported in my paper, and run python3 event_chain.py to train a PairLSTM model, and run python3 main_with_args.py to train a SGNN model. I have writen some annotations in my codes, please read them and run!

It is a very time consuming extraction pipeline from the raw NYT/Gigaword raw corpus to get the preprocessed data. The good news is that you don't need to download the Gigaword corpus, because I have provided all the data you need to run the code.

## Requirements
* python 3.6
* PyTorch 0.3.0
* Tesla P100

