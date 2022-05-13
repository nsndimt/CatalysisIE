
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


### Download annotated data, pretrained BERT and NER model checkpoint
Download [dataset](https://zenodo.org/record/6533264#.YnqkmhPMKdY) from zenodo and follow the dataset instruction to extract the data

### Reproduce
The `reproduce.ipynb` notebook provides instructions to reproduce our main expriment result using ALL data

### Prection
The `train.ipynb` notebook gives examples of how to train your own model on your own data set

### Prection
The `prediction.ipynb`  notebook gives examples of how to use our model to extract catalysis information on given text

### Application
We are sorry that we cannot distribute the whole search engine and correlation analysis system since some articles in our collection are not open-accessed. Sharing those is against [Elsevier's TDM policy](https://www.elsevier.com/about/policies/text-and-data-mining/text-and-data-mining-faq). But we provide the key piece of code of our system, which should help you build a similar system on you own data.

The catalysis search engine can be accessed through: infochain.ece.udel.edu/catalysis_search. The catalysis correlation analysis can be accessed through: infochain.ece.udel.edu/catalysis_correlation

#### search engine (`streamlit_correlation.py`)
Here we show you how we search query related articles from our highly relevant article collection
###### Pyserini
follow the `Guide to indexing and searching English documents` section at pyserini's [guideline](https://github.com/castorini/pyserini/#sparse-indexes)
1. we split each article into paragraphs using sliding window algorithm: the 1st paragraph contains the 1st to 10th sentence, the 2nd paragraph contains the 11th to 20th sentence and so on. As lessons learned from [TREC-COVID challenge](https://github.com/castorini/anserini/blob/master/docs/experiments-cord19.md) and [TREC 2007 genomics track](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-46) shows that treat the full text as a single document is not a good design. A window size of 10 is proven to be effective on [MS MARCO Document Ranking](https://arxiv.org/pdf/2101.05667.pdf)
2. we use BM25(k1=1.2, b=0.9) to search paragraphs, we tune the parameters of BM25 by optimizing the following retrieval task: we using the title of each article as query and search the whole collection of article abstracts. The higher we rank the corresponding abstract which comes from the same article as the title, the better the parameters. This is the same as what is done in [Content-Based Weak Supervision for Ad-Hoc Re-Ranking
](https://arxiv.org/abs/1707.00189)
3. The final rank is based on the sum of the top 3 paragraph BM25 scores of each article. Note that directly take the maxiumal score is more popular as suggested by expriments on [Robust04](https://arxiv.org/pdf/1905.09217.pdf). We use top 3 mainly for better visual display of articles.

###### Pymongo
Here we store the metadata of each paper in MongoDB and use Pymongo to access it. Feel free change `load_db` and `fetch_paper` function to use other metadata storage.

#### correlation analysis system( `streamlit_correlation.py`)
Here we show you how we analyze the correlation between entities
1. you need to provide a pandas DataFrame which contains `doi`,`sent_i`,`label`,`span` and `norm_span` five columns. It should covers all extracted chemical entities, their index(which paper, which sentence), their label(which type of entity), their normalization form(mainly to aggregate alternative expressions)
2. Currently, we search the normalized entity, which could cause the following problems: `Ru on C` returns no result, since `Ru on C` will be normlized into `Ru C`. So searching `Ru C` has result while searching `Ru on C` has no result. One alternative is to search the raw entity, but in this way, the co-occurance patter will be greatly weakened by various alternative expression. So we decided to design the system in current form. In the furture, we may have better normalization techniques to solve this issue. 



### Contact
Please create an issue or email to [zhangyue@udel.edu](mailto:zhangyue@udel.edu) should you have any questions.
