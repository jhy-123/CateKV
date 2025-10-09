# [ICML 2025] CateKV
This is the official repository of "CateKV: On Sequential Consistency for Long-context LLM Acceleration"

[Paper](https://openreview.net/pdf?id=u7dlwgKstN)

## Environment Setup
For Conda users, you can directly create the environment using our provided configuration file:

```bash
conda env create -f catekv.yml
conda activate catekv
```

If you prefer using pip, please refer to the environment setup guide from [ShadowKV](https://github.com/ByteDance-Seed/ShadowKV) for detailed installation steps and dependency management.

## Download models
The pretrained models supported in our experiments can be downloaded and placed in the `ckpt` directory.  

We currently provide several supported models for convenience:
- Llama-3-8B-1M: [gradientai/Llama-3-8B-Instruct-Gradient-1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)
- Phi-3-Mini-128K: [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- Llama-3.1-8B: [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Yi-9B-200K: [01-ai/Yi-9B-200K](https://huggingface.co/01-ai/Yi-9B-200K)

## Headmask Generation (Optional)

If you are using one of the **supported models**, we provide the pre-generated headmasks in the `select_headmask` directory.  
These can be used directly, and you may **skip this step**.

If you wish to use a **new model** or **re-generate the headmask**, we provide a script to do so:

```bash
bash run_headmask.sh
```

For the reference dataset, we use the Variable Tracking task from the RULER benchmark by default. You may replace it with another task if desired. The goal of this step is to determine the optimal percentile-based threshold k and scaling factor α for head selection.

After selecting the optimal parameters, the final headmask can be produced using:
```bash
python select_headmask/static.py
```

## Accuracy Evaluations
We provide the accuracy evaluation of **CateKV** on the [RULER](https://github.com/hsiehjackson/RULER) benchmark.

### Building the Dataset
To prepare the RULER dataset, please run the following commands:
```bash
# prepare RULER dataset
python -c "import nltk; nltk.download('punkt')"
cd data/ruler
bash create_dataset.sh "gradientai/Llama-3-8B-Instruct-Gradient-1048k" "llama-3"
```

### Run Evaluations
To reproduce the accuracy evaluation for both the baseline and **CateKV**, please run the following commands with a single A100 GPU:
```bash
bash run_baseline.sh
bash run_catekv.sh
```

## Efficiency Evaluations
For the efficiency evaluation, please run the following command with a single A100 GPU:

```bash
python test/eval_efficiency.py --model_name "ckpt/Llama-3-8B-Instruct-Gradient-1048k" --head_classification_path "select_headmask/llama3" --datalen 512000 --method "CateKV" --cv_threshold 0.4 --full_fraction 1.0
```

## Citation
If you find **CateKV** useful for your research or development, please cite the following:

```bibtex
@inproceedings{jiangcatekv,
  title={CateKV: On Sequential Consistency for Long-Context LLM Inference Acceleration},
  author={Jiang, Haoyun and Huang, Fei and Hu, Qiang and Sun, Minmin and Xiao, Shuai and Li, Yong and Lin, Junyang and Yao, Jiangchao and others},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Acknowledgements
This project has been influenced by many excellent projects, such as [ShadowKV](https://github.com/ByteDance-Seed/ShadowKV) and others.
