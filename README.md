## Requirements

- Python 3.6
- TensorFlow 1.6

## Instructions

- To train and evaluate the model, run the following cmd from root of the project:
```bash
python train.py
```

- To use the model to predict a sentiment for custom input:
```bash
python predict.py --text <your-text-in-quotes>
```

## A short report on the classification task

### Dataset Stats

Class-wise Distribution

Positive - 2363 (16.14%)  
Negative - 9178 (62.69%)  
Neutral - 3099 (21.16%)  

Total : 14640

### Data Analysis

- Only processing that I did on the tweets is to convert them from unicode to ascii so that each emoji in the dataset has its char-level representation captured.

Additional data processing that could improve the model:

- Usernames can sometimes impart (irrelevant) sentiment towards the overall sentiment of the tweet. For example, if ‘@UnitedAirways’ occurs in majority of the negative tweets then it could have more weight in the final sentiment of the tweet being inclined towards negative. Overall sentiment of a tweet should have no relation to the usernames being mentioned in the tweet in a generic approach for a classifier. They can be removed and tested for accuracy and precision.
- Links in the form or `http:/*` or `t.co/*` would not help in predicting the sentiment. Hence, they can be removed.
- I have used both lowercase and uppercase letters in my final data because there usually is a emotion attached when people use uppercase letters so I experimented by keeping them. They could all be converted to lowercase (like in most of the applications) and tested as well.

### Result Analysis

- Best accuracy obtained was 63.28% which is not good at all considering that a simple majority baseline classifier choosing negative sentiment would have given around 62%.
- The network implemented here is an RNN with a basic cell mostly due to lack of time for finding a better architecture and my lack of dexterity with Tensorflow for certain tasks. Results can be further improved (or not) by using LSTM, Bi-LSTM or CNN. CNNs are found to outperform RNNs in text classification, especially sentiment analysis, as [studied by Kim Yoon](https://arxiv.org/abs/1408.5882) and more recently in [sequence modeling tasks](https://arxiv.org/abs/1803.01271). A CNN network could be character embedding layer followed by convolutional layer, a max-pooling layer and a softmax layer at the end similar to the one used by Kim Yoon.
- Changing from character level to word level embeddings and leveraging Word2Vec or GloVe vectors could improve results on this dataset.
- There is, however, a certain amount of bias in the dataset with negative class covering 62% of the dataset.
- [Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) for initializing weights is one other thing I’d try.
- More analysis could have been done if I had calculated confusion matrix, or the preferred metrics for classification tasks - precision, recall and f1-score.

### Training

- Note that I am not using batches here since the dataset is very small but having batches is always good for extensibility of the code.
- The paths to save and restore the models from can be fed via parameters to the script and their naming could be dynamic based on the time training was done but I have simply defined the paths and the names for the model in the config to save some time on my part.
- The optimizer I have used is SGD with momentum due to the minima found by it being usually better for character-level embeddings than, say, Adam optimizer which is however known to have better convergence. I experimented with both of them but there wasn’t much of a performance difference on this dataset.

### Codebase

- Codebase could be further improved by creating a models directory and having one file/class per architecture. In this project, I have created just one class (BasicRNN) which can further co-exist alongside other files/classes for different networks so that they can be easily used in `train.py` as required which makes the code extensible.
- `util.py` can be moved to a new directory ‘utils’ to have a clean separation of concern for utility related functions.
- It is good to have all the generic configurations which are model/network agnostic inside one file as done here using `config.py`.
- Having only `train.py` and `predict.py` in the root of the folder would make a good way to go about structuring the code in my opinion as far as simple classifiers with different networks are considered.
- A simple directory structure can be as shown below:
```
data/
networks/ # one class per architecture
 - BasicRNN.py
 - BasicLSTM.py
models/ #save models here
train_test.py (train and testing could be separated too)
Predict.py
```
- BasicRNN class that I have created in `models.py` could be rewritten and structured by
following the guidelines mentioned here .