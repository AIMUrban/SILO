## SILO
This is the **PyTorch implementation** of [**SILO: Semantic Integration for Location Prediction with Large Language Models**](https://doi.org/10.1145/3711896.3737129).

### Datasets

For access to **raw datasets**, please refer to the [Humob Challenge 2024](https://wp.nyu.edu/humobchallenge2024/). Follow the instructions on their website for data access and usage policies.

According to the paper's definition, we processed the raw dataset to extract user activity locations. This resulted in `train_sampled.parquet` and `test_sampled.parquet` files. Both files contain the following five attributes, consistent with the original dataset: `uid`, `d`, `t`, `x`, `y`.


### Large Language Model
We use **GPT-2** as the backbone model with **LoRA fine-tuning**. The base models can be accessed from [Hugging Face](https://huggingface.co/openai-community/gpt2).  
For efficient fine-tuning, we recommend using **LoRA for PyTorch**, which can be found [here](https://github.com/microsoft/LoRA).

### Running Steps
* Optional arguments can be modified in config.py.
1. **Preprocess text-based semantics using LLMs**
   ```bash
   python semantic_prompt.py --gpu 0,1,2,3 --dataset data
2. **Run the model**
   ```bash
   python main.py --gpu 0,1,2,3 \
              --batch 128 \
              --epoch 20 \
              --loc_dim 128 --time_dim 128 --user_dim 768 \
              --dataset data --llm gpt2 \
              --lora ---verbose
   
### Citation
```bibtex
@inproceedings{sun2025silo,
  title={SILO: Semantic Integration for Location Prediction with Large Language Models},
  author={Sun, Tianao and Chen, Meng and Zhang, Bowen and Dai, Genan and Huang, Weiming and Zhao, Kai},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
