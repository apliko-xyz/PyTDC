<p align="center"><img src="https://raw.githubusercontent.com/apliko-xyz/PyTDC/master/pytdc_logo.png" alt="logo" width="600px" /></p>

----
[![website](https://img.shields.io/badge/website-live-brightgreen)](https://pytdc.apliko.io)
[![PyPI version](https://img.shields.io/pypi/v/pytdc-nextml.svg)](https://pypi.org/project/pytdc-nextml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/pytdc-nextml.svg)](https://pypi.org/project/pytdc-nextml/)
[![Downloads](https://pepy.tech/badge/pytdc/month)](https://pepy.tech/project/pytdc)
[![Downloads](https://pepy.tech/badge/pytdc)](https://pepy.tech/project/pytdc)
[![GitHub Repo stars](https://img.shields.io/github/stars/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/network/members)

![Conda Github Actions Build](https://github.com/apliko-xyz/PyTDC/actions/workflows/conda-tests.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


[**ICML 2025 Paper**](https://openreview.net/forum?id=HV8vZDDoYc) | [**NeurIPS 2024 AIDrugX Paper**](https://openreview.net/forum?id=kL8dlYp6IM)

# Introducing PyTDC
Existing biomedical benchmarks do not provide end-to-end infrastructure for training, evaluation, and inference of models that integrate multimodal biological data and a broad range of machine learning tasks in therapeutics. We present PyTDC, an open-source machine-learning platform providing streamlined training, evaluation, and inference software for multimodal biological AI models. PyTDC unifies distributed, heterogeneous, continuously updated data sources and model weights and standardizes benchmarking and inference endpoints.

The components of PyTDC include:

- A collection of multimodal, continually updated heterogeneous data sources is unified under the introduced "API-first-dataset" architecture. Inspired by **API-first design**, this microservice architecture uses the model-view-controller design pattern to enable multimodal data views.
- PyTDC presents open-source model retrieval and deployment software that streamlines AI inferencing and exposes **state-of-the-art, research-ready models** and training setups for biomedical representation learning models across modalities.
- We integrate **single-cell analysis with multimodal machine learning in therapeutics** via the introduction of contextualized tasks.

<p align="center"><img src="https://github.com/mims-harvard/TDC/blob/12be2b9f5ab39480d5489cf3867126f41287598b/fig/TDCneurips.pptx(1).png" alt="workflow" width="600px" /></p>
We present PyTDC, a machine-learning platform providing streamlined training, evaluation, and inference software for single-cell biological foundation models to accelerate research in transfer learning method development in therapeutics. PyTDC introduces an API-first architecture that unifies heterogeneous, continuously updated data sources. The platform introduces a model server, which provides unified access to model weights across distributed repositories and standardized inference endpoints. The model server accelerates research workflows by exposing state-of-the-art, research-ready models and training setups for biomedical representation learning models across modalities. Building upon Therapeutic Data Commons, we present single-cell therapeutics tasks, datasets, and benchmarks for model development and evaluation.

[**\[Learn More\]**](https://tdcommons.ai/pytdc)

## Built on the Therapeutics Data Commons (TDC)

PyTDC has forked from Therapeutics Data Commons, a datasets store with ml-ready-datasets in 66 tasks- https://github.com/mims-harvard/TDC

## Key PyTDC Presentations and Publications

[0] Western Bioinformatics Seminar Series: Alejandro Velez-Arce, "Signals in the Cells: Multimodal and Contextualized Machine Learning Foundations for Therapeutics." [**\[Event\]**](https://www.events.westernu.ca/events/schulich-medicine-dentistry/2024-11/western-bioinformatics-nov14.html) [**\[Seminar\]**](https://western-bioinfo.github.io/seminars/alejandro-velez-arce) [**\[Slides\]**](https://neurips.cc/media/neurips-2024/Slides/102832.pdf)

[1] Velez-Arce, Huang, Li, Lin, et al., Signals in the Cells: Multimodal and Contextualized Machine Learning Foundations for Therapeutics, NeurIPS AIDrugX, 2024 [**\[Paper\]**](https://openreview.net/pdf?id=kL8dlYp6IM) [**\[Slides\]**](https://neurips.cc/media/neurips-2024/Slides/102832.pdf) [**\[Webpage\]**](https://tdcommons.ai/pytdc)


## Installation

### Using `pip`

To install the core environment dependencies of TDC, use `pip`:

```bash
pip install pytdc-nextml
```



## Cite Us

If you find PyTDC useful, cite our [ICML paper](https://openreview.net/forum?id=HV8vZDDoYc) and [NeurIPS paper](https://openreview.net/pdf?id=kL8dlYp6IM):

```
@inproceedings{
velez-arce2025pytdc,
title={Py{TDC}: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models},
author={Alejandro Velez-Arce and Marinka Zitnik},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=HV8vZDDoYc}
}
```

```
@inproceedings{
velez-arce2024signals,
title={Signals in the Cells: Multimodal and Contextualized Machine Learning Foundations for Therapeutics},
author={Alejandro Velez-Arce and Xiang Lin and Kexin Huang and Michelle M Li and Wenhao Gao and Bradley Pentelute and Tianfan Fu and Manolis Kellis and Marinka Zitnik},
booktitle={NeurIPS 2024 Workshop on AI for New Drug Modalities},
year={2024},
url={https://openreview.net/forum?id=kL8dlYp6IM}
}
```


PyTDC is built on top of other open-sourced projects. Additionally, please cite the original work if you used these datasets/functions in your research. You can find the original paper for the function/dataset on the website. For older datasets, please cite the paper:

```
@article{Huang2021tdc,
  title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley,
          Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  journal={Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks},
  year={2021}
}
```


## Data Server

Many PyTDC datasets are hosted on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/21LKWG) with the following persistent identifier [https://doi.org/10.7910/DVN/21LKWG](https://doi.org/10.7910/DVN/21LKWG). When Dataverse is under maintenance, PyTDC datasets cannot be retrieved. That happens rarely; please check the status on [the Dataverse website](https://dataverse.harvard.edu/).

## License
The PyTDC codebase is licensed under the MIT license. For individual dataset usage, please refer to the dataset license on the website.
