"""Measure per-layer weight effective rank directly from pretrained checkpoints.

For each transformer block, pool squared singular values from its Linear weights,
then compute spectral-entropy effective rank. No examples or training data are used.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from layer_selection import transformer_blocks

MODELS = {
    'Pythia-2.8B': 'EleutherAI/pythia-2.8b',
    'Llama-2-7B': 'meta-llama/Llama-2-7b-hf',
    'GPT-Neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
}


def spectral_rank(weights, device):
    # A block-diagonal concatenation has the union of component singular values.
    # Squared singular values match the spectral-energy effective-rank definition.
    energy = []
    max_rank = 0
    for weight in weights:
        matrix = weight.detach().to(device=device, dtype=torch.float32)
        singular = torch.linalg.svdvals(matrix)
        energy.append(singular.square())
        max_rank += min(matrix.shape)
        del matrix, singular
    values = torch.cat(energy)
    values = values[values > 0]
    probabilities = values / values.sum()
    rank = torch.exp(-(probabilities * probabilities.log()).sum()).item()
    return rank, max_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float32')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    records=[]; metadata={'rank_definition':'exp entropy of pooled squared singular values',
        'objects':'all torch.nn.Linear weight matrices within each transformer block',
        'aggregation':'block-diagonal spectral union; normalized by sum of component min(out,in)',
        'data_used':None,'dtype_for_svd':'float32','models':{}}
    for label in args.models:
        model_id=MODELS[label]
        print(f'Loading {label} weights',flush=True)
        model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=dtype,
             low_cpu_mem_usage=True,device_map=None)
        blocks=transformer_blocks(model)
        layer_records=[]
        for idx,block in enumerate(blocks):
            weights=[module.weight for module in block.modules()
                     if isinstance(module,nn.Linear) and module.weight.ndim==2]
            if not weights:
                raise RuntimeError(f'{label} layer {idx}: no Linear weights found')
            rank,max_rank=spectral_rank(weights,device)
            row=dict(model=label,model_id=model_id,layer=idx,
                effective_rank=rank,max_component_rank=max_rank,
                normalized_rank=rank/max_rank,matrix_count=len(weights))
            layer_records.append(row);records.append(row)
            print(f'  layer {idx}: effective_rank={rank:.1f}/{max_rank} ({rank/max_rank:.3f})',flush=True)
        metadata['models'][label]=dict(model_id=model_id,num_layers=len(blocks),layers=layer_records)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
    with (output/'all_models.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'weight_ranks.json').write_text(json.dumps(metadata,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    colors=['#3478A4','#D17A32','#27856A']
    for (label,item),color in zip(metadata['models'].items(),colors):
        rows=item['layers'];x=[r['layer'] for r in rows]
        axes[0].plot(x,[r['effective_rank'] for r in rows],marker='o',ms=2.8,lw=1.5,label=label,color=color)
        axes[1].plot(x,[r['normalized_rank'] for r in rows],marker='o',ms=2.8,lw=1.5,label=label,color=color)
    for ax,title,ylabel in [(axes[0],'Layer weight effective rank','Effective rank'),
                            (axes[1],'Normalized layer weight rank','Effective rank / maximum component rank')]:
        ax.set_title(title);ax.set_xlabel('Transformer block');ax.set_ylabel(ylabel)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(output/'layer_weight_ranks.pdf',bbox_inches='tight')
    fig.savefig(output/'layer_weight_ranks.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (output/'README.md').write_text(
        '# Layer weight effective-rank profiles\n\n'
        'These curves are computed from each pretrained checkpoint’s layer Linear weight matrices; no text data is read. '
        'Within a block, squared singular values from all Linear weights are pooled and effective rank is the exponential entropy. '
        'The normalized curve divides by the sum of each matrix’s maximum possible rank. '
        'This is a block-level summary of multiple matrices, not the activation effective rank or a single-matrix theorem quantity. '
        'Compare the profile shape with the layerwise pruning memorization curve as a hypothesis test; similarity is not assumed. '
        'SVD cost depends on model size and device.\n')


if __name__=='__main__': main()
