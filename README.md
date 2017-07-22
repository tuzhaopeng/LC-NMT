# LC-NMT
Larger-Context NMT

NMT-Coverage
===========================

In this version, we introduce a coverage mechanism (NMT-Coverage) to indicate whether a source word is translated or not, which proves to alleviate over-translation and under-translation. If you use the code, please cite <a href="http://arxiv.org/abs/1601.04811">our paper</a>:

<pre><code>@InProceedings{Tu:2016:ACL,
  author    = {Wang, Longyue and Tu, Zhaopeng and Way, Andy and Liu Qun},
  title     = {Exploiting Cross-Sentence Context for Neural Machine Translation},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  year      = {2017},
}
</code></pre>

For any comments or questions, please  email <a href="mailto:vincentwang0229@gmail.com">Longyue Wang</a> and <a href="mailto:tuzhaopeng@gmail.com">Zhaopeng Tu</a>.


Installation
------------

NMT-Coverage is developed by <a href="http://www.zptu.net">Zhaopeng Tu</a>, which is on top of lisa-groudhog (https://github.com/lisa-groundhog/GroundHog). It requires Theano0.8 or above version (for the module "scan" used in the trainer).

To install NMT-Coverage in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------



