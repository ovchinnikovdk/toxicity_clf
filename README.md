# Toxic Comment Classification Challenge
![Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

Models: 
- CNN Model `models.cnn.TextCNN`
- BiLSTM with Avg pooling `models.bi_lstm.BiLSTM`
- BiLSTM with Attention `models.bi_lstm_attn.BiLSTM_Attn`
- Stacking models `models.stacking.StackingModels`

Best score for now: 0.96 Public LeaderBoard with Single model (BiLSTM).

## Training

For train run:
```shell script
python train.py --model=model_name --n_epochs=N
```

Loss: `BCEWithLogits`

Optimizer: `Adam` with `ReduceLROnPlateau`
 
Metrics: AUC, F1-score. 

## Inference 

For inference run:
```shell script
python infer.py --model=model_name --path=PATH_TO_CSV_FILE
```

## Vocabulary
Simple Dict vocabulary with top 10k frequent words, excluding stopwords. 
For additional information please see: `vocab.CommentVocab`.

## Links
Inspired by 

![Kaggle 1](https://www.kaggle.com/abhinav2308/pytorch-toxic-comment-solution)

![Kaggle 2](https://www.kaggle.com/yekenot/textcnn-2d-convolution)

![Kaggle 3](https://www.kaggle.com/fizzbuzz/bi-lstm-conv-layer-lb-score-0-9840)

![Kaggle 4](https://www.kaggle.com/ybonde/cleaning-word2vec-lstm-working)

![Github Text Classification](https://github.com/prakashpandey9/Text-Classification-Pytorch)

![Doc2Vec](https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-doc2vec)


 