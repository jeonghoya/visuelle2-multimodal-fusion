import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# [GFLOPS 측정을 위한 라이브러리]
try:
    from thop import profile
except ImportError:
    print("Please install thop: pip install thop")
    profile = None

from dataset_fusion import Visuelle2
# [핵심 수정] 학습시킨 M4FT 모델 클래스 임포트
# (파일명을 models/M4FT_Visuelle2.py 로 저장하셨다고 가정합니다. 
# 만약 models/Proposed_model.py 로 저장하셨다면 그에 맞게 수정하세요.)
try:
    from models.M4FT_Visuelle2 import M4FT_Visuelle2 as M4FTModel
except ImportError:
    # 혹시 파일명이 다를 경우를 대비한 예외처리 (사용자 환경에 맞게 수정 가능)
    from models.M4FT_Visuelle2 import M4FT_Visuelle2 as M4FTModel

def run(args):
    print(f"=== Forecasting M4FT (Dual-Gated) on Task: {'Demand' if args.demand else 'SO-fore'} (Output Len: {args.output_len}) ===")
    
    # 1. Load Data Info
    test_df = pd.read_csv(
        os.path.join(args.dataset_path, "stfore_test.csv"),
        parse_dates=["release_date"],
    )
    cat_dict = torch.load(os.path.join(args.dataset_path, "category_labels.pt"))
    col_dict = torch.load(os.path.join(args.dataset_path, "color_labels.pt"))
    fab_dict = torch.load(os.path.join(args.dataset_path, "fabric_labels.pt"))
    gtrends = pd.read_csv(
        os.path.join(args.dataset_path, "vis2_gtrends_data.csv"), index_col=[0], parse_dates=True
    )

    demand = bool(args.demand)
    img_folder = os.path.join(args.dataset_path, 'images')
    vis_pt_test = "visuelle2_test_processed_demand.pt" if demand else "visuelle2_test_processed_stfore.pt"

    # 2. Dataset
    # 학습 때와 동일하게 autoregressive 옵션 전달
    testset = Visuelle2(
        sales_df=test_df,
        img_root=img_folder,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=52,
        demand=demand,
        local_savepath=os.path.join(args.dataset_path, vis_pt_test),
        output_len=args.output_len,
        autoregressive=bool(args.autoregressive) 
    )
    
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test batches: {len(testloader)}")

    # 3. Load Model
    print(f"Loading Checkpoint: {args.ckpt_path}")
    
    model = M4FTModel.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_len,
        num_heads=4, 
        num_layers=args.num_layers,      
        use_text=True, use_img=True,
        cat_dict=cat_dict, col_dict=col_dict, fab_dict=fab_dict, store_num=125,
        trend_len=52, num_trends=3, gpu_num=args.gpu_num,
        autoregressive=bool(args.autoregressive),
        strict=False # 유연한 로딩을 위해 False 권장
    )
    
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 4. Prediction Loop
    gt_list, forecast_list = [], []
    gflops_val = 0.0
    
    for i, batch in enumerate(tqdm(testloader, desc="Forecasting")):
        data_tuple, images = batch
        
        images = images.to(device)
        data_tuple = [d.to(device) for d in data_tuple]

        # Task에 따른 데이터 분기 (0-Padding 로직 포함)
        if len(data_tuple) == 8:
            item_sales, y, category, color, fabric, store, temporal_features, gtrends = data_tuple
        else: # Demand Task (No History)
            y, category, color, fabric, store, temporal_features, gtrends = data_tuple
            bs = y.shape[0]
            # M4FT 모델도 Demand Task에서는 Sales 입력 자리에 0을 넣어야 함
            item_sales = torch.zeros(bs, 1, 2, device=device)

        # [GFLOPS 측정]
        if i == 0 and profile is not None:
            try:
                inputs = (item_sales, category, color, fabric, store, temporal_features, gtrends, images)
                macs, params = profile(model, inputs=inputs, verbose=False)
                
                total_gflops = (2 * macs) / 1e9
                gflops_val = total_gflops / item_sales.shape[0] 
                
                print(f"\n[Profile] Batch Size: {item_sales.shape[0]}")
                print(f"[Profile] Total MACs: {macs}")
                print(f"[Profile] Total GFLOPs (Batch): {total_gflops:.4f}")
                print(f"[Profile] GFLOPs per Sample: {gflops_val:.4f}\n")
            except Exception as e:
                print(f"\n[Warning] Failed to measure GFLOPS: {e}\n")

        with torch.no_grad():
            forecast, _ = model(item_sales, category, color, fabric, store, temporal_features, gtrends, images)
        
        gt_list.append(y.contiguous().view(-1))
        forecast_list.append(forecast.contiguous().view(-1))

    # 5. Metrics
    norm_scalar = 53.0
    gt = torch.cat(gt_list).cpu().numpy() * norm_scalar
    forecasts = torch.cat(forecast_list).cpu().numpy() * norm_scalar
    
    mae = np.mean(np.abs(gt - forecasts))
    wape = 100 * np.sum(np.abs(gt - forecasts)) / np.sum(np.abs(gt))
    
    print(f"\n=== Final Results ===")
    print(f"WAPE:   {wape:.4f} %")
    print(f"MAE:    {mae:.4f}")
    if profile is not None:
        print(f"GFLOPS: {gflops_val:.4f} (per sample)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../visuelle2/')
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_num", type=int, default=0)
    
    parser.add_argument("--demand", type=int, default=0, help="1=Demand, 0=SO-fore")
    parser.add_argument("--output_len", type=int, default=1, help="Length of Prediction")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers used in training")
    parser.add_argument('--autoregressive', type=int, default=0) 
    
    parser.add_argument("--embedding_dim", type=int, default=32) 
    parser.add_argument("--hidden_dim", type=int, default=64)
    
    args = parser.parse_args()
    
    # Demand Task일 때만 강제로 12로 설정
    if args.demand: args.output_len = 12
    
    run(args)