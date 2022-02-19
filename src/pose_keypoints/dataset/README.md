# Place the dataset here

## Dataset Directory Structure

```None
dataset/
            |->images/
            |        000000.jpg
            |        000001.jpg
            |        000002.jpg
            |->labels/
            |        000000.json
            |        000001.json
            |        000002.json
            |->labels.json
            |->training_set.txt
```

* All images must be in .jpg
* labels.json: Should contain the full dataset in COCO Structure
* training_set.txt: Should contain the filenames for the training set
* (Optional) validation_set.txt
* (Optional) testing_set.txt