# Trainable parameters

Trainable parameter counts for full-network updates, targeted STUDE (all block parameters in zero-indexed layers 10–24, inclusive), and LoRA-STUDE (rank 16, attention Q/K/V in the same layers; fused QKV for Pythia). Counts come from configuration-only meta-device models and actual PEFT adapter insertion; tied weights are counted once. Frozen base weights remain resident for LoRA. Counts describe gradient-update scope, not total model storage, runtime, optimizer memory, or the number of nonzero weights. The preliminary pruning edit is excluded. Full-network is a reference update scope, not a measured DEL or SPE budget.

| Model | Full network | Targeted | LoRA | Targeted % | LoRA % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pythia-2.8B | 2,775,208,960 | 1,180,147,200 | 2,457,600 | 42.52 | 0.089 |
| Llama-2-7B | 6,738,415,616 | 3,035,750,400 | 5,898,240 | 45.05 | 0.088 |
| GPT-Neo-2.7B | 2,651,307,520 | 1,180,032,000 | 3,686,400 | 44.51 | 0.139 |

Configuration sources:
- [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b/raw/main/config.json)
- [Llama-2-7B](https://huggingface.co/docs/transformers/model_doc/llama2)
- [GPT-Neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B/raw/main/config.json)
