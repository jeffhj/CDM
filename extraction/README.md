# CDM (Extraction)



---

## SDI-Extractor ##

All the following scripts need to be run in order.

### Train the Classifier ###
```python3 pretain.py```

* This step will split the dataset into train:valid:test = 8:1:1

```python3 def_classifer.py```

* This step will train the classifier using the dataset splitted as stated above.
* The best model will be saved in `./def_data/Model/`

### Collect Candidate Definitions from Web ###
```python3 kw_crawler.py```

* Candidate definitions of terms in `Keywords-Springer-83K-20210405.csv` will be collected and stored.
* All the plain texts with URLs will be stored in `./full_text/` 
* All the candidate definitions will be stored in `./has_kw_sent/`

### Definition Prediction ###
```python3 pred.py```

* Use the classifier to predict candidate definitions.
* Predicted classes and confidence scores will be stored in `./predout/`



---

## CDI-Extractor

You may use [Wikipedia API](# https://pypi.org/project/wikipedia/) to collect CDI:

- Use `wikipedia.search(term)` to collect related terms

- Use `wikipedia.summary(term)` to collect summaries of terms
- extract first sentences in the summaries

```
# https://pypi.org/project/wikipedia/
import wikipedia
def get_wiki_search_result(term, mode=0):
    if mode==0:
        return wikipedia.search(f"\"{term}\"")
    else:
        return wikipedia.search(term)
```



