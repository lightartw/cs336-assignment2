import os
import logging
from typing import Dict, List, Tuple
import re
import glob
import torch 
import typer
import json
import pandas as pd

from cs336_systems.benchmark_model import ModelBenchmarker
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
    model_sizes: List[str] = typer.Option(["small", "medium"], help="--model-sizes small --model-sizes medium"),
    context_lengths: List[int] = typer.Option([128, 256], help="上下文长度列表"),
    config_path: str = typer.Option("./cs336_systems/model_config.json", help="配置文件路径"),
    precision: str = typer.Option("bf16", help=" fp32, fp16, bf16"),
    only_forward: bool = typer.Option(False),
    num_warmups: int = typer.Option(5, help="预热步数"),
    num_trials: int = typer.Option(10, help="测量步数"),
    use_nvtx: bool = typer.Option(False, help="是否使用nsys"),
    profile_mem: bool = typer.Option(False),
    use_compile: bool = typer.Option(False),
):
    batch_size = 4
    vocab_size = 10000

    if not os.path.exists(config_path):
        logger.error(f"can't find config: {config_path}")
        raise typer.Exit(code=1)
        
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    if use_compile:
        config_dict['training']['is_compile'] = True
    
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
                    use_nvtx=use_nvtx,
                    profile_memory=profile_mem
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

@app.command()
def bench_atten(
    config_path: str = typer.Option("./cs336_systems/model_config.json", help="配置文件路径"),
    num_warmups: int = 10,
    num_trials: int = 100,
    is_compile: bool = typer.Option(False, "--is-compile", help="是否使用 torch.compile"),
):
    import itertools
    from cs336_systems.benchmark_atten import AttentionBenchmarker
    from cs336_basics.model import scaled_dot_product_attention

    atten_fn = scaled_dot_product_attention
    if is_compile:
        logger.info("Using torch.compile")
        torch.set_float32_matmul_precision('high')
        atten_fn = torch.compile(scaled_dot_product_attention)

    batch_size = 8
    dmodels =[16, 32, 64, 128]
    seq_lens =[256, 1024, 4096]

    benchmarker = AttentionBenchmarker(config_path)

    results_list = []
    for dmodel, seq_len in itertools.product(dmodels, seq_lens):
        logger.info(f"\nBenchmarking dmodel={dmodel}, seq_len={seq_len}")
        res_entry = {
            "d_model": dmodel,
            "seq_len": seq_len,
            "Forward (ms)": "N/A",
            "Backward (ms)": "N/A",
            "Memory (GB)": "N/A"
        }   

        try:
            shape = (batch_size, seq_len, dmodel)
            Q = torch.randn(*shape, device="cuda", requires_grad=True)
            K = torch.randn(*shape, device="cuda", requires_grad=True)
            V = torch.randn(*shape, device="cuda", requires_grad=True)
            #mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).bool()
            mask = None 

            res = benchmarker.benchmark_attention_step(
                atten_fn,
                Q, K, V, mask, 
                num_warmups=num_warmups, num_trials=num_trials
            )
            
            if res.get("oom", False):
                error_msg = "OOM"
                res_entry["Forward (ms)"] = error_msg
                res_entry["Backward (ms)"] = error_msg
                res_entry["Memory (GB)"] = error_msg
            else:
                res_entry["Forward (ms)"] = f"{res['forward_mean_ms']:.2f} ± {res['forward_std_ms']:.2f}"
                res_entry["Backward (ms)"] = f"{res['backward_mean_ms']:.2f} ± {res['backward_std_ms']:.2f}"
                res_entry["Memory (GB)"] = f"{res['memory_gb']:.4f} ± {res['memory_std']:.4f}"
            
            del Q, K, V, mask
            
        except Exception as e:
            error_msg = "Error"
            if "out of memory" in str(e).lower():
                error_msg = "OOM"
            logger.error(f"Failed at d={dmodel}, l={seq_len}: {e}")
            res_entry["Forward (ms)"] = error_msg
            res_entry["Backward (ms)"] = error_msg
            res_entry["Memory (GB)"] = error_msg
        finally:
            results_list.append(res_entry)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print(f" 📊 Attention Benchmark Results (Naïve Implementation) | Batch Size: {batch_size}")
    print("="*80 + "\n")
    
    print(df.to_markdown(index=False))
    print("\n" + "="*80)

@app.command()
def bench_flash(
    warmups: int=25,
    trails: int=100,
    use_flash: bool = typer.Option(True, help="是否使用flash"),
    precision: str = typer.Option("bf16", help="fp32, bf16") 
):
    import itertools
    from cs336_systems.benchmark_atten import benchmark_flash
    from cs336_basics.model import scaled_dot_product_attention
    from cs336_systems.flashattention import FlashAttention2Triton

    atten_fn = scaled_dot_product_attention
    if use_flash:
        atten_fn = FlashAttention2Triton.apply
    else:
        def baseline_wrapper(q, k, v, is_causal=True):
            seq_len = q.shape[-2]
            if is_causal:
                mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            else:
                mask = None
            return scaled_dot_product_attention(q, k, v, mask)
        atten_fn = baseline_wrapper

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.bfloat16)
    n_heads = 16
    dmodels = [16, 32, 64, 128]
    #seq_lens = [128, 256, 512, 1024, 2048, 4096]
    seq_lens = [8192]

    logger.info(f"dtype: {dtype}")
    results_list = []
    for dmodel, seq_len in itertools.product(dmodels, seq_lens):
        logger.info(f"Benchmarking dmodel={dmodel}, seq_len={seq_len}")
        res_entry = {
            "d_model": dmodel,
            "seq_len": seq_len,
            "Forward (ms)": "N/A",
            "Backward (ms)": "N/A",
            "End-to-End (ms)": "N/A"
        }   

        try:
            fwd_ms, bwd_ms, e2e_ms = benchmark_flash(
                n_heads=n_heads, 
                seq_len=seq_len, 
                d_model=dmodel, 
                dtype=dtype, 
                attn_fn=atten_fn,
                warmups=warmups,
                trials=trails
            )
            res_entry["Forward (ms)"] = f"{fwd_ms:.2f}"
            res_entry["Backward (ms)"] = f"{bwd_ms:.2f}"
            res_entry["End-to-End (ms)"] = f"{e2e_ms:.2f}"
            
        except Exception as e:
            error_msg = "Error"
            logger.error(f"Failed at d={dmodel}, l={seq_len}: {e}")
            res_entry["Forward (ms)"] = error_msg
            res_entry["Backward (ms)"] = error_msg
            res_entry["End-to-End (ms)"] = error_msg
        finally:
            results_list.append(res_entry)

    df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print(f" 📊 FlashAttention Benchmark Results")
    print("="*80 + "\n")
    
    print(df.to_markdown(index=False))
    print("\n" + "="*80)

if __name__ == "__main__":
    app()