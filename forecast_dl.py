import os
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from dataset import Visuelle2
from models.CrossAttnRNN21 import CrossAttnRNN as Model21
from models.CrossAttnRNN210 import CrossAttnRNN as Model210
from models.CrossAttnRNNDemand import CrossAttnRNN as DemandModel
from tqdm import tqdm
from utils import calc_error_metrics

# [GFLOPS 측정을 위한 라이브러리]
try:
    from thop import profile
except ImportError:
    print("Please install thop: pip install thop")
    profile = None

def run(args):
    print(f"=== Forecasting Task: {'Demand' if args.new_product else 'SO-fore (2-10)'} ===")
    print(args)

    # Seed for reproducibility
    pl.seed_everything(args.seed)

    ####################################### Load data #######################################
    test_df = pd.read_csv(
        os.path.join(args.dataset_path, "stfore_test.csv"),
        parse_dates=["release_date"],
    )

    # Load attribute encodings
    cat_dict = torch.load(os.path.join(args.dataset_path, "category_labels.pt"))
    col_dict = torch.load(os.path.join(args.dataset_path, "color_labels.pt"))
    fab_dict = torch.load(os.path.join(args.dataset_path, "fabric_labels.pt"))

    # Load Google trends
    gtrends = pd.read_csv(
        os.path.join(args.dataset_path, "vis2_gtrends_data.csv"), index_col=[0], parse_dates=True
    )

    demand = bool(args.new_product)
    img_folder = os.path.join(args.dataset_path, 'images')

    visuelle_pt_test = "visuelle2_test_processed_demand.pt" if demand else "visuelle2_test_processed_stfore.pt"

    # [수정 1] Dataset에 output_len 전달
    testset = Visuelle2(
        test_df,
        img_folder,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        52,
        demand,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_test),
        output_len=args.output_len 
    )

    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    print(f"Test batches: {len(testloader)}")

    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    
    ####################################### Load Model #######################################
    # Load model structure first
    model = None
    if demand:
        model = DemandModel(
            attention_dim=args.attention_dim,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_trends=3,
            cat_dict=cat_dict, 
            col_dict=col_dict, 
            fab_dict=fab_dict,
            store_num=125,
            use_img=True, 
            use_att=True, 
            use_date=True,
            use_trends=True,
            out_len=12
        )
    else:
        if args.task_mode == 0:
            model = Model21(
                attention_dim=args.attention_dim,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                use_img=args.use_img,
                out_len=args.output_len,
            )
        else:
            # [수정 2] 2-10 모델 초기화 시 메타데이터 전달 (Train과 동일하게)
            print("Initializing Full-Feature 2-10 Model...")
            model = Model210(
                attention_dim=args.attention_dim,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                cat_dict=cat_dict, 
                col_dict=col_dict, 
                fab_dict=fab_dict,
                store_num=125,
                num_trends=3,
                use_img=args.use_img,
                out_len=args.output_len,
                use_teacher_forcing=False, # 추론 시에는 Teacher Forcing 끔
                teacher_forcing_ratio=0.0,
            )

    # Load Weights
    print(f"Loading weights from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    gt, forecasts = [], []
    gflops_val = 0.0

    for i, data in enumerate(tqdm(testloader, desc="Forecasting")):
        
        # Unpack Data
        (data_tuple, images) = data
        images = images.to(device)
        
        # [수정 3] 추론 시 모든 입력값 언패킹 및 전달
        if demand:
            ts, categories, colors, fabrics, stores, temporal_features, gtrends = [d.to(device) for d in data_tuple]
            model_inputs = (ts, categories, colors, fabrics, stores, temporal_features, gtrends, images)
            y = ts 
        else:
            # 2-10 모델도 이제 모든 피처를 받음
            X, y, category, color, fabric, store, temporal_features, gtrends = [d.to(device) for d in data_tuple]
            
            if args.task_mode == 0: # Model 2-1 (기존 방식 유지 필요 시 수정)
                 model_inputs = (X, y, images)
            else: # Model 2-10 (Full Feature)
                 model_inputs = (X, y, category, color, fabric, store, temporal_features, gtrends, images)
            
        # [GFLOPS 측정]
        if i == 0 and profile is not None:
            try:
                macs, params = profile(model, inputs=model_inputs, verbose=False)
                total_gflops = (2 * macs) / 1e9
                gflops_val = total_gflops / y.shape[0] 
                
                print(f"\n[Profile] Batch Size: {y.shape[0]}")
                print(f"[Profile] Total MACs: {macs}")
                print(f"[Profile] Total GFLOPs (Batch): {total_gflops:.4f}")
                print(f"[Profile] GFLOPs per Sample: {gflops_val:.4f}\n")
            except Exception as e:
                print(f"\n[Warning] Failed to measure GFLOPS: {e}\n")

        # Inference
        with torch.no_grad():
            if demand:
                y_hat, _, _ = model(*model_inputs)
            else:
                if args.task_mode == 0:
                     y_hat, _ = model(*model_inputs)
                else:
                     # 2-10 Full Feature
                     y_hat, _ = model(*model_inputs)

        forecasts.append(y_hat.contiguous().cpu())
        gt.append(y.contiguous().cpu())

    # Metrics
    # Visuelle 2.0 베이스라인은 norm_scalar 파일 로드 대신 고정값 53.0을 쓰기도 함
    # 파일이 있다면 로드, 없으면 53.0 사용
    try:
        norm_scalar = np.load(os.path.join(args.dataset_path, 'stfore_sales_norm_scalar.npy'))
    except:
        norm_scalar = 53.0

    gt_tensor = torch.cat(gt).squeeze()
    forecasts_tensor = torch.cat(forecasts).squeeze()
    
    # Rescale
    gt_rescaled = gt_tensor.numpy() * norm_scalar
    forecasts_rescaled = forecasts_tensor.numpy() * norm_scalar
    
    mae = np.mean(np.abs(gt_rescaled - forecasts_rescaled))
    wape = 100 * np.sum(np.abs(gt_rescaled - forecasts_rescaled)) / np.sum(np.abs(gt_rescaled))

    print(f"\n=== Final Results ===")
    print(f"WAPE:   {wape:.4f} %")
    print(f"MAE:    {mae:.4f}")
    if profile is not None:
        print(f"GFLOPS: {gflops_val:.4f} (per sample)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../visuelle2/')
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--new_product", type=int, default=1, help="1=Demand, 0=SO-fore")
    
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_len", type=int, default=1) # Command line argument overrides this
    parser.add_argument("--use_img", type=bool, default=True)
    parser.add_argument("--task_mode", type=int, default=0, help="0-->2-1 - 1-->2-10")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--use_teacher_forcing", action='store_true')
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.3)

    parser.add_argument("--ckpt_path", type=str, required=True)
    
    args = parser.parse_args()
    
    # Demand는 12, 2-10은 10으로 자동 설정 (인자가 없으면)
    if args.new_product:
        args.output_len = 12
    elif args.task_mode == 1 and args.output_len == 1: # Default값인 경우
        args.output_len = 10
        
    run(args)