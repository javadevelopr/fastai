# Fastai: 2 - Image classification: Production
## Demo
### Electric cars classifier
A CNN classifier to predict classes of popular electric cars.
Uses Streamlit for a simple user interface. 
User should be able to enter an image url and get a predicted label, if the image contains an electric car.

#### To Run:
```
pip install --upgrade torch torchvision fastai streamlit

streamlit run app.py
```

#### If you have [conda/miniconda](https://docs.conda.io/en/latest/) installed:

```
conda env create -f environment.yml
```

Follow the instructions then activate the conda environment:

```
conda activate javadev865-streamlit-fastai-env
```

And finally run the app:

```
streamlit run app.py
```
