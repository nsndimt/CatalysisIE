
### Installation required Python package

1. pytorch==1.8.1
2. pytorch-lightning==1.5.10
2. transformers==4.6.1
3. tqdm
4. numpy
5. pandas
5. seqeval
6. stanza==1.2.0
7. streamlit==1.1.0
8. pyserini==0.13.0


### Download pretrained BERT and NER model checkpoint
Download [pretrained BERT](https://drive.google.com/file/d/1guDHCs6Y_ybORQJem5pEqHmtZ3ZlQrsc/view?usp=sharing) and [model checkpoints](https://drive.google.com/file/d/19yKMp0vEeKaaHXjWtnKgoml-NOx1h9LE/view?usp=sharing), extract it using `tar xvzf`, and put everything into `pretrained` and `checkpoint` directory, respectively

### Reproduce
The `reproduce.ipynb` notebook provides instructions to reproduce our main expriment result using ALL data

### Prection
The `train.ipynb` notebook gives examples of how to train your own model on your own data set

### Prection
The `prediction.ipynb`  notebook gives examples of how to use our model to extract catalysis information on given text

### Application
The `application.ipynb`  notebook gives guideline on how to reuse our system code and build your own application on your own data

### Contact
Please create an issue or email to [zhangyue@udel.edu](mailto:zhangyue@udel.edu) should you have any questions.
