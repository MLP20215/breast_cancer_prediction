# Requirements for Data

## Where to get

Both CAMELYON16 and CAMELYON17 data sets can be downloaded at the [official website](https://camelyon17.grand-challenge.org/data/).

## Data usage in our project

| Dataset type | Data source | Number of patients | Number of WSIs per patient | Number of total WSIs |        Description        |
|:------------:|:-----------:|:------------------:|----------------------------|:--------------------:|:-------------------------:|
| Training set |  CAMELYON17 |         60         |              5             |          300         | From center 1 to center 3 |
|   Test set   |  CAMELYON16 |         10         |              1             |          10          |      Randomly chosen      |

## Directory structure

```txt
CAMELYON16
|-----training
|     |----lesion_annotations           # It is okay to leave these directories empty
|     |----normal                       # because we did not used training data
|     |----tumor                        # from CAMELYON16.
|
|-----test
|     |----lesion_annotations
|     |----images
|
|
CAMELYON17
|-----training
|     |----center_0
|     |----center_1
|     |----center_2
|     |----lesion_annotations
```
