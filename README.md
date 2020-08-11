# How to Trust Your Deep Learning Code

This is the support repository for the blog post [How to Trust Your Deep Learning Code](http://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html).
It contains code for training a Variational Autoencoder (VAE) and the associated unit tests.
The unit tests illustrate useful concepts to test in deep learning projects.
The focus lay on writing tests that are readable and reusable.

For more information check out the blog post.

## Usage

The project uses Python 3.7.
First, install the packages specified in the `requirements.txt` file.

```
conda create -n unittest_dl python=3.7
conda activate unittest_dl
conda install --file requirements.txt

## or

virtualenv -p python3.7 unittest_dl
source unittest_dl/bin/activate
pip install -r requirements.txt
```

This project was developed in PyCharm, so it is the easiest to use that way.
Open it in the IDE and mark the `src` directory as Sources Root (right-click the folder > Mark directory as > Sources Root).
Everything should work out of the box now.
To run all tests, right-click the `tests` directory and select `"Run 'Unittests in tests'"`.

As an alternative, you can manually add `src` to your `PYTHON_PATH` environment variable.