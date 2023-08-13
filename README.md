# What Constitutes Good Contrastive Learning in Time-Series Forecasting?
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of Informer in the following paper: 
[What Constitutes Good Contrastive Learning in Time-Series Forecasting?](https://arxiv.org/abs/2306.12086). We developed our code based on the repositories of [Informer](https://github.com/zhouhaoyi/Informer2020) and [CoST](https://github.com/salesforce/CoST). 

This repo implements the experiments of end-to-end training For end-to-end two-step training experiments, please refer to [this repo](https://github.com/chiyuzhang94/contrastive_learning_time-series_two-phase)

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following paper:

```
@article{DBLP:journals/corr/abs-2306-12086,
  author       = {Chiyu Zhang and
                  Qi Yan and
                  Lili Meng and
                  Tristan Sylvain},
  title        = {What Constitutes Good Contrastive Learning in Time-Series Forecasting?},
  journal      = {CoRR},
  volume       = {abs/2306.12086},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2306.12086},
  doi          = {10.48550/arXiv.2306.12086},
  eprinttype    = {arXiv},
  eprint       = {2306.12086},
  timestamp    = {Fri, 23 Jun 2023 15:19:11 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2306-12086.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact
If you have any questions, feel free to contact Chiyu Zhang through Email (zcy94@outlook.com) or Github issues. 
## Acknowledgments
We acknowledge the authors of the repositories of [Informer](https://github.com/zhouhaoyi/Informer2020) and [CoST](https://github.com/salesforce/CoST). 