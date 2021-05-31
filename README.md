# Dimensionwise Separable 2-D Graph Convolution for Unsupervised and Semi-Supervised Learning on Graphs

This is a TensorFlow implementation of DSGC (a 2-D graph filtering framework) for semi-supervised and unsupervised classification proposed in our paper:

* Qimai LI, Xiaotong Zhang, Han Liu, Quanyu Dai, and Xiao-Ming Wu. 2021. Dimensionwise Separable 2-D Graph Convolution for Unsupervisedand Semi-Supervised Learning on Graphs. In Proceedings of *the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’21), August 14–18, 2021, Virtual Event, Singapore*. ACM, New York, NY, USA, 12 pages.

## Requirements
Our code is tested with packages of following version:
* Python=3.7
* TensorFlow=1.14
* numpy=1.18.1
* cupy=7.5.0 (optional)
## Download Dataset
Go to https://www.dropbox.com/s/r4btgq1djubgfwt/DSGC-KDD21.zip?dl=0, download all files in `DSGC-KDD21/data/` directory and put them into your local `data/` directory. 

## Reproduce Experimental Results

### Classification
Run `train.py` to reproduce classification results. Edit `config.py` to change settings, models, and parameters. You can set 'repeat' to 1 for fast try.

```bash
# 20 labels/class
python train.py --seed 0 --XF --GXF --train-size 20 --weight-decay 5e-5 --repeat 50 --dataset large_cora
python train.py --seed 0 --XF --GXF --train-size 20 --weight-decay 5e-6 --repeat 50 --dataset 20news
python train.py --seed 0 --XF --GXF --train-size 20 --weight-decay 5e-7 --repeat 50 --dataset wiki

# 5 labels/class
python train.py --seed 0 --XF --GXF --train-size 5  --weight-decay 5e-5 --repeat 50 --dataset large_cora
python train.py --seed 0 --XF --GXF --train-size 5  --weight-decay 5e-6 --repeat 50 --dataset 20news
python train.py --seed 0 --XF --GXF --train-size 5  --weight-decay 5e-7 --repeat 50 --dataset wiki

# WebKB
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_cornell
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_texas
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_wisconsin
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_washington
```

### Clustering
Run `clustering.py` to reproduce clustering results of `20 Newgroups`. Edit `clustering.py` to change dataset.

```bash
python clustering.py
```

## Acknowledgement

This project is largely based on Thomas N. Kipf's GCN [code](https://github.com/tkipf/gcn). Thanks kipf for making code public. The source of dataset is listed here:
* Cora: [Full 40k+ Paper Dataset](https://people.cs.umass.edu/~mccallum/data.html) can be found here. The [11881 paper subset](https://sites.google.com/site/semanticbasedregularization/home/software/experiments_on_cora) we used was selected by Marco Gori. We did not use features he provided, but process raw papers by ourselves instead.
* Wikispeedia: https://snap.stanford.edu/data/wikispeedia.html.
* 20 Newgroups: http://qwone.com/~jason/20Newsgroups/
* WebKB: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-11/www/wwkb/

Thanks them for providing first-hand data.
