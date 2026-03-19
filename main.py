import os
import logging
from typing import List

import torch 
import typer
import json
import pandas as pd

from cs336_systems.benchmarking import ModelBenchmarker
from cs336_systems.config import Config

app = typer.Typer(help="CS336 Benchmarking", add_completion=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_SPECS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@app.command()
def sweep(
    model_sizes: List[str] = typer.Option(["small", "medium", "large"], help="--model-sizes small --model-sizes medium"),
    context_lengths: List[int] = typer.Option([128, 256], help="上下文长度列表"),
    config_path: str = typer.Option("./cs336_systems/model_config.json", help="配置文件路径"),
    vocab_size: int = typer.Option(10000, help="always 10,000"),
    batch_size: int = typer.Option(4, help="always 4"),
    precision: str = typer.Option("fp32", help=" fp32, fp16, bf16"),
    only_forward: bool = typer.Option("True"),
    num_warmups: int = typer.Option(5, help="预热步数"),
    num_trials: int = typer.Option(10, help="测量步数"),
):
    if not os.path.exists(config_path):
        logger.error(f"can't find config: {config_path}")
        raise typer.Exit(code=1)
        
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    logger.info(f"🚀 Start Benchmarking Sweep | 精度: {precision}")
    results =[]

    for size_name in model_sizes:
        if size_name not in MODEL_SPECS:
            logger.warning(f"Unknown Size {size_name}, skip...")
            continue
            
        spec = MODEL_SPECS[size_name]
        
        config_dict['model'].update({
            'd_model': spec['d_model'],
            'd_ff': spec['d_ff'],
            'num_layers': spec['num_layers'],
            'num_heads': spec['num_heads'],
            'vocab_size': vocab_size
        })
        config_dict['training'].update({
            'batch_size': batch_size,
            'precision': precision
        })
        
        for ctx_len in context_lengths:
            logger.info(f"Testing Model={size_name}, Context={ctx_len}...")
            config_dict['model']['context_length'] = ctx_len
            
            res_entry = {
                "Model Size": size_name,
                "Context Length": ctx_len,
                "Mode": "Forward Only" if only_forward else "Full Step"
            }

            try:
                config_obj = Config.model_validate(config_dict)
                benchmarker = ModelBenchmarker(config_obj)
                
                mean, std = benchmarker.benchmark_step(
                        only_forward=only_forward, num_warmups=num_warmups, num_trials=num_trials
                )
                res_entry["Time (ms)"] = f"{mean:.2f} ± {std:.2f}"
                res_entry["Time (ms)"] = f"{mean:.2f} ± {std:.2f}"
                results.append(res_entry)
                
            except RuntimeError as e:
                error_msg = "OOM / Error"
                logger.error(f"评测 {size_name} (ctx: {ctx_len}) 时出错: {e}")
                res_entry["Time (ms)"] = error_msg
                results.append(res_entry)
                
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not results:
        logger.error("未收集到任何测试结果。")
        raise typer.Exit(code=1)

    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print(f" 📊 Benchmark Results | Precision: {precision} | Batch: {batch_size}")
    print("="*60 + "\n")
    
    print(df.to_markdown(index=False))
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    app()