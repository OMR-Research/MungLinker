# munglinker

Experiments on linking the nodes of a music notation graph (MuNG).

## Requirements

- Python 3.6
- [PyTorch](https://pytorch.org/)

Install with ``pip install -e .`` to follow updates without having to re-install.

Download [MUSCIMA++ dataset](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2372) and extract the xml-files from `data/cropobjects_withstaff` into `data/mungs`.

Don't download [CVC-MUSCIMA](http://www.cvc.uab.es/cvcmuscima/index_database.html) dataset, but just download and extract the used images from [here](https://owncloud.tuwien.ac.at/index.php/s/Xv91caXnPubL6Zk/download) into `data/images`

Run the training by calling:

`python munglinker/train.py`