# LC-NMT
Larger-Context Neural Machine Translation

Larger-Context NMT v0.1
===========================

In this version, we introduce a cross-sentence context-aware approach to investigate the influence of historical contextual information on the performance of NMT. If you use the code, please cite <a href="https://arxiv.org/pdf/1704.04347.pdf">our paper</a>:

<pre><code>@InProceedings{Wang:2017:EMNLP,
  author    = {Wang, Longyue and Tu, Zhaopeng and Way, Andy and Liu Qun},
  title     = {Exploiting Cross-Sentence Context for Neural Machine Translation},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  year      = {2017},
}
</code></pre>

This package is based on the old version of Nematus by EdinburghNLP (https://github.com/EdinburghNLP/nematus). If you use Nematus, please cite the following paper:

<pre><code>@InProceedings{sennrich-EtAl:2017:EACLDemo,
  author    = {Sennrich, Rico  and  Firat, Orhan  and  Cho, Kyunghyun  and  Birch, Alexandra  and  Haddow, Barry  and  Hitschler, Julian  and  Junczys-Dowmunt, Marcin  and  L\"{a}ubli, Samuel  and  Miceli Barone, Antonio Valerio  and  Mokry, Jozef  and  Nadejde, Maria},
  title     = {Nematus: a Toolkit for Neural Machine Translation},
  booktitle = {Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {65--68},
  url       = {http://aclweb.org/anthology/E17-3017}
}
</code></pre>

For any comments or questions, please  email <a href="mailto:vincentwang0229@gmail.com">Longyue Wang</a> and <a href="mailto:tuzhaopeng@gmail.com">Zhaopeng Tu</a>.

Installation
------------

LC-NMT is developed by <a href="http://computing.dcu.ie/~lwang/">Longyue Wang</a> and <a href="http://www.zptu.net">Zhaopeng Tu</a>, which is on top of Nematus. It requires Theano0.8 or above version (for the module "scan" used in the trainer).

To install LC-NMT in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------

1, Pre-processing

We need to process train/dev/test files from common format into historical one. Assume that the number of history sentence is 2 (hist_len = 2). For example:

The orignal format is:

<pre><code>And of course , we all share the same adaptive imperatives .
We 're all born . We all bring our children into the world .
We go through initiation rites .
We have to deal with the inexorable separation of death , so it shouldn 't surprise us that we all sing , we all dance , we all have art .</code></pre>

The historical format is:

<pre><code>NULL@@@@And of course , we all share the same adaptive imperatives .
And of course , we all share the same adaptive imperatives .@@@@We 're all born . We all bring our children into the world .
And of course , we all share the same adaptive imperatives .####We 're all born . We all bring our children into the world .@@@@We go through initiation rites .
We 're all born . We all bring our children into the world .####We go through initiation rites .@@@@We have to deal with the inexorable separation of death , so it shouldn 't surprise us that we all sing , we all dance , we all have art .</code></pre>

where the delimiter "@@@@" is used to separate historical and current sentences; and the the delimiter "####" is sentence boundary.

We use jieba segmentor (https://github.com/fxsjy/jieba) for Chinese segmentation, and use Moses scripts (https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer) for English tokenization.

2, Training

2.1 Use data_iterator_src_hist.py instead of data_iterator.py for loading data.

2.2 Use hnmt_src_static_*.py instead of nmt.py for training. For example, hnmt_src_static_gate_src_init_two.py means +Init_{enc+dec}+Gating Auxi in the paper.

2.3 Use translate_src_static_*.py instead of translate.py for decoding. For instance, translate_src_static_gate_src_init_two.py means +Init_{enc+dec}+Gating Auxi in the paper.

TO-do List
--------------------------

1, re-code based on new version of Nematus;

2, release other codes for variant models in the paper.

