import os
import logging
from typing import Dict, List, Tuple
import re
import glob
import torch 
import typer
import json
import pandas as pd

from cs336_systems.benchmark import ModelBenchmarker
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
    model_sizes: List[str] = typer.Option(["small"], help="--model-sizes small --model-sizes medium"),
    context_lengths: List[int] = typer.Option([128], help="上下文长度列表"),
    config_path: str = typer.Option("./cs336_systems/model_config.json", help="配置文件路径"),
    vocab_size: int = typer.Option(10000, help="always 10,000"),
    batch_size: int = typer.Option(4, help="always 4"),
    precision: str = typer.Option("fp32", help=" fp32, fp16, bf16"),
    only_forward: bool = typer.Option("False"),
    num_warmups: int = typer.Option(5, help="预热步数"),
    num_trials: int = typer.Option(10, help="测量步数"),
    use_nvtx: bool = typer.Option(False, help="是否使用nsys")
):
    if not os.path.exists(config_path):
        logger.error(f"can't find config: {config_path}")
        raise typer.Exit(code=1)
        
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    logger.info(f"🚀 Start Benchmarking Sweep | Precision: {precision}")
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
                
                res = benchmarker.benchmark_step(
                    only_forward=only_forward, 
                    num_warmups=num_warmups, 
                    num_trials=num_trials,
                    use_nvtx=use_nvtx
                )
                mean, std = res['step']

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

@app.command()
def report(
    csv_dir: str = typer.Option("./result/csv", help="nsys 导出的 CSV 文件所在目录")
):
    registry = {
        "Forward": ":forward",
        "Backward": ":backward",
        "Optimizer": ":optimizer_step"
    }

    raw_storage: Dict[Tuple[str, int], Dict[str, Tuple[float, float]]] = {}
    
    pattern = os.path.join(csv_dir, "profile_*_ctx*_nvtx_sum.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        logger.error(f"在 {csv_dir} 中未找到匹配的 CSV 文件。")
        raise typer.Exit(code=1)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        match = re.search(r"profile_(.*)_ctx(\d+)_nvtx_sum", filename)
        if not match: continue
        model_size, ctx_len = match.group(1), int(match.group(2))
        
        key = (model_size, ctx_len)
        if key not in raw_storage: raw_storage[key] = {}

        try:
            df = pd.read_csv(file_path)
            col = 'Name' if 'Name' in df.columns else 'Range'
            
            for title, label in registry.items():
                row = df[df[col].str.contains(label, na=False)]
                if not row.empty:
                    raw_storage[key][title] = (row['Avg (ns)'].values[0] / 1e6, 
                                               row['StdDev (ns)'].values[0] / 1e6)
        except Exception as e:
            logger.error(f"解析 {filename} 失败: {e}")

    size_order = {"small": 0, "medium": 1, "large": 2}
    def print_formatted_table(title: str, data_list: List[dict]):
        if not data_list: return
        df_res = pd.DataFrame(data_list)
        df_res['order'] = df_res['Model Size'].map(lambda x: size_order.get(x, 99))
        df_res = df_res.sort_values(by=['order', 'Context Length']).drop(columns=['order'])
        print(f"\n {title} Pass")
        print(df_res.to_markdown(index=False))

    for title in registry.keys():
        table_rows = []
        for (size, ctx), metrics in raw_storage.items():
            res = metrics.get(title)
            val = f"{res[0]:.2f} ± {res[1]:.2f}" if res else "N/A"
            table_rows.append({"Model Size": size, "Context Length": ctx, "Time (ms)": val})
        print_formatted_table(title, table_rows)

    total_rows = []
    for (size, ctx), metrics in raw_storage.items():
        if all(t in metrics for t in registry.keys()):
            t_med = sum(m[0] for m in metrics.values())
            t_std = sum(m[1]**2 for m in metrics.values())**0.5 
            val = f"{t_med:.2f} ± {t_std:.2f}"
        else:
            val = "OOM / Incomplete"
        total_rows.append({"Model Size": size, "Context Length": ctx, "Time (ms)": val})
    
    print_formatted_table("Total Step (F+B+O)", total_rows)



if __name__ == "__main__":
    app()