# TextGAN-PyTorch

TextGAN is a PyTorch framework for Generative Adversarial Networks (GANs) based text generation models. TextGAN serves as a benchmarking platform to support research on GAN-based text generation models. Since most GAN-based text generation models are implemented by Tensorflow, TextGAN can help those who get used to PyTorch to enter the text generation field faster.

For now, only few GANs-based models are implemented, including [SeqGAN (Yu et. al, 2017)](https://arxiv.org/abs/1609.05473), [LeakGAN (Guo et. al, 2018)](https://arxiv.org/abs/1709.08624) and [RelGAN (Nie et. al, 2018)](https://openreview.net/forum?id=rJedV3R5tm). If you find any mistake in my implementation, please let me know! Also, please feel free to contribute to this repository if you want to add other models.



## Requirement

- PyTorch >= 1.0.0
- Python 3.6
- Numpy 1.14.5
- CUDA 7.5+ (For GPU)
- nltk 3.4



## Implemented Models and Original Papers

- **SeqGAN** - [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)
- **LeakGAN** - [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624)
- **RelGAN** - [RelGAN: Relational Generative Adversarial Networks for Text Generation](https://openreview.net/forum?id=rJedV3R5tm)



## Get Started

- Get Started

```bash
git clone https://github.com/williamSYSU/TextGAN-PyTorch.git
cd TextGAN-PyTorch
```

- Run with <code>SeqGAN</code>

```bash
cd run
python3 run_seqgan.py
```

- Run with <code>LeakGAN</code>

```bash
cd run
python3 run_leakgan.py
```

- Run with <code>RelGAN</code>

```bash
cd run
python3 run_relgan.py
```



## TODO

- [ ] Add Experiment Results
- [ ] Fix bugs in <code>LeakGAN</code> model
- [ ] Add instructors of <code>SeqGAN</code> and <code>LeakGAN</code> in <code>instrutor/real_data</code>