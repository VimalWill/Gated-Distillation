"""Count trainable parameters on meta-device models; no weights or GPU required.

Reads the architecture-selection helpers directly from train_model.py's AST to
avoid its unrelated dataset/sklearn imports while matching its current behavior.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import torch
import transformers
import peft
from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoConfig, LlamaConfig
from peft import LoraConfig, get_peft_model


def training_helpers():
    source = Path(__file__).with_name('train_model.py').read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('get_layer_prefix', 'lora_targets')]
    assert len(nodes) == 2
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'train_model.py', 'exec'), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def configurations():
    return [
        ('Pythia-2.8B', GPTNeoXConfig(hidden_size=2560, intermediate_size=10240,
            num_hidden_layers=32, num_attention_heads=32, vocab_size=50304,
            max_position_embeddings=2048, tie_word_embeddings=False),
         'https://huggingface.co/EleutherAI/pythia-2.8b/raw/main/config.json'),
        ('Llama-2-7B', LlamaConfig(hidden_size=4096, intermediate_size=11008,
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
            vocab_size=32000, max_position_embeddings=4096, tie_word_embeddings=False,
            attention_bias=False, mlp_bias=False),
         'https://huggingface.co/docs/transformers/model_doc/llama2'),
        ('GPT-Neo-2.7B', GPTNeoConfig(hidden_size=2560, intermediate_size=10240,
            num_layers=32, num_heads=20, vocab_size=50257,
            max_position_embeddings=2048, attention_types=[[['global', 'local'], 16]],
            tie_word_embeddings=True),
         'https://huggingface.co/EleutherAI/gpt-neo-2.7B/raw/main/config.json'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--layer-start', type=int, default=10)
    parser.add_argument('--layer-end', type=int, default=24)
    parser.add_argument('--output', type=Path, default=Path('figures/parameter_budget'))
    args = parser.parse_args()
    if args.rank < 1 or not 0 <= args.layer_start <= args.layer_end < 32:
        parser.error('Require rank > 0 and 0 <= layer-start <= layer-end < 32')
    helpers, digest = training_helpers()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for name, config, source in configurations():
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        assert all(p.is_meta for p in model.parameters())
        full = sum(p.numel() for p in model.parameters())
        prefix = helpers['get_layer_prefix'](model)
        selected = []
        for parameter_name, parameter in model.named_parameters():
            trainable = prefix in parameter_name and args.layer_start <= int(
                parameter_name.split(prefix)[1].split('.')[0]) <= args.layer_end
            parameter.requires_grad_(trainable)
            if trainable:
                selected.append(dict(name=parameter_name, count=parameter.numel()))
        targeted = sum(p.numel() for p in model.parameters() if p.requires_grad)
        targets, pattern = helpers['lora_targets'](model)
        projections = []
        for module_name, module in model.named_modules():
            if (prefix in module_name and module_name.split('.')[-1] in targets
                and args.layer_start <= int(module_name.split(prefix)[1].split('.')[0]) <= args.layer_end):
                projections.append(dict(name=module_name, input=module.in_features, output=module.out_features))
        expected = sum(args.rank*(p['input']+p['output']) for p in projections)
        with torch.device('meta'):
            model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank,
                lora_dropout=0, target_modules=targets, layers_pattern=pattern,
                layers_to_transform=list(range(args.layer_start, args.layer_end+1)),
                bias='none', task_type='CAUSAL_LM'))
        lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert lora == expected > 0, (name, lora, expected)
        assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
        assert all(p.is_meta for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == full+lora
        row = dict(model=name, full=full, targeted=targeted, lora=lora,
                   targeted_percent=100*targeted/full, lora_percent=100*lora/full,
                   full_to_lora=full/lora, targeted_to_lora=targeted/lora)
        rows.append(row)
        details.append(dict(model=name, config=config.to_dict(), config_source=source,
                            targeted_tensors=selected, lora_projections=projections))
        print(json.dumps(row), flush=True)
        del model
    with (args.output/'counts.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'audit.json').write_text(json.dumps(dict(rank=args.rank,
        layer_start=args.layer_start, layer_end=args.layer_end, train_model_sha256=digest,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
        counts=rows, models=details),indent=2)+'\n')
    caption = (f'Trainable parameter counts for full-network updates, targeted STUDE '
        f'(all block parameters in zero-indexed layers {args.layer_start}–{args.layer_end}, inclusive), '
        f'and LoRA-STUDE (rank {args.rank}, attention Q/K/V in the same layers; fused QKV for Pythia). '
        'Counts come from configuration-only meta-device models and actual PEFT adapter insertion; '
        'tied weights are counted once. Frozen base weights remain resident for LoRA. '
        'Counts describe gradient-update scope, not total model storage, runtime, optimizer memory, '
        'or the number of nonzero weights. The preliminary pruning edit is excluded. '
        'Full-network is a reference update scope, not a measured DEL or SPE budget.')
    md = ['# Trainable parameters', '', caption, '',
          '| Model | Full network | Targeted | LoRA | Targeted % | LoRA % |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(f"| {r['model']} | {r['full']:,} | {r['targeted']:,} | {r['lora']:,} | {r['targeted_percent']:.2f} | {r['lora_percent']:.3f} |")
    md += ['', 'Configuration sources:'] + [f"- [{d['model']}]({d['config_source']})" for d in details]
    (args.output/'README.md').write_text('\n'.join(md)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.0,3.4),layout='constrained')
    for j,(key,label,color) in enumerate([
        ('full','Full network','#687886'),('targeted','Targeted STUDE','#3478A4'),
        ('lora',f'LoRA-STUDE (r={args.rank})','#D17A32')]):
        positions=[i+(j-1)*0.25 for i in range(3)]
        vals=[r[key] for r in rows]
        ax.bar(positions, vals, width=0.23, color=color, label=label,zorder=3)
        for x,v in zip(positions,vals):
            ax.text(x,v*1.18,f'{v/1e9:.2f}B' if v>=1e9 else f'{v/1e6:.2f}M',ha='center',fontsize=8)
    ax.set_yscale('log'); ax.set_ylim(1e6,4e10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v/1e9:g}B' if v>=1e9 else f'{v/1e6:g}M'))
    ax.set_xticks(range(3), [r['model'] for r in rows])
    ax.set_ylabel('Trainable parameters (log scale)')
    ax.grid(axis='y',which='major',alpha=0.2,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.19),ncol=3,frameon=False,fontsize=8)
    for ext in ['pdf','svg','png']:
        fig.savefig(args.output/f'parameter_budget.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (args.output/'figure.tex').write_text('\\begin{figure*}[t]\n\\centering\n'
        '\\includegraphics[width=0.9\\textwidth]{figures/parameter_budget/parameter_budget.pdf}\n'
        '\\caption{'+caption.replace('–','--')+'}\n\\label{fig:parameter-budget}\n\\end{figure*}\n')


if __name__ == '__main__':
    main()
