import os
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime
import time

# [Visuelle 2.0 데이터셋]
from dataset_fusion import Visuelle2 
# [핵심 수정] M4FT 모델 임포트
from models.Proposed_model_v3 import TARG_M4FT_Visuelle2 as MyModel

def run(args):
    print("=== Training M4FT (Dual-Gated) on Visuelle 2.0 ===")
    print(args)

    # Seed for reproducibility
    pl.seed_everything(args.seed)

    ####################################### Load Data #######################################
    # 1. Load CSVs
    train_df = pd.read_csv(
        os.path.join(args.dataset_path, "stfore_train.csv"),
        parse_dates=["release_date"],
    )
    test_df = pd.read_csv(
        os.path.join(args.dataset_path, "stfore_test.csv"),
        parse_dates=["release_date"],
    )

    # 2. Load Attribute Encodings
    cat_dict = torch.load(os.path.join(args.dataset_path, "category_labels.pt"))
    col_dict = torch.load(os.path.join(args.dataset_path, "color_labels.pt"))
    fab_dict = torch.load(os.path.join(args.dataset_path, "fabric_labels.pt"))

    # 3. Load Google Trends
    gtrends = pd.read_csv(
        os.path.join(args.dataset_path, "vis2_gtrends_data.csv"), index_col=[0], parse_dates=True
    )

    # 4. Task Setup (Demand vs SO-fore)
    demand = bool(args.demand)
    img_folder = os.path.join(args.dataset_path, 'images')
    
    # 데이터셋 파일 설정
    if demand:
        args.output_len = 12 # Demand Task는 12주 고정
        visuelle_pt_train = "visuelle2_train_processed_demand.pt"  
        visuelle_pt_test = "visuelle2_test_processed_demand.pt"  
    else:
        # 2-1 / 2-10 Task
        visuelle_pt_train = "visuelle2_train_processed_stfore.pt"
        visuelle_pt_test = "visuelle2_test_processed_stfore.pt"

    # 5. Create Datasets
    trainset = Visuelle2(
        sales_df=train_df,
        img_root=img_folder,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=52,        
        demand=demand,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_train),
        output_len=args.output_len,
        autoregressive=bool(args.autoregressive)
    )
    testset = Visuelle2(
        sales_df=test_df,
        img_root=img_folder,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=52,        
        demand=demand,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_test),
        output_len=args.output_len,
        autoregressive=bool(args.autoregressive)
    )

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Dataset Loaded. Train batches: {len(trainloader)}, Test batches: {len(testloader)}")


    ####################################### Model Setup #######################################
    # [수정] M4FT 모델 초기화
    # 인자는 GTM과 동일하게 유지 (호환성)
    model = MyModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_len,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        use_text=args.use_text,
        use_img=args.use_img,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        store_num=125,   
        trend_len=52,
        num_trends=3,
        gpu_num=args.gpu_num,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive
    )

    ####################################### Training #######################################
    
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    
    # [수정] 체크포인트 파일명 변경 (M4FT)
    task_name = "Demand" if demand else f"SO-{args.output_len}"
    filename_format = f"Gated_method-{task_name}-epoch={{epoch}}-{dt_string}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=filename_format,
        monitor="val_wWAPE", 
        mode="min",
        save_top_k=1,
    )

    if args.use_wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run
        )

    trainer = pl.Trainer(
        gpus=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger if args.use_wandb else None,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5 
    )

    start_time = time.time()
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)
    end_time = time.time()

    print(f"[Training Completed] Time: {(end_time - start_time)/60:.2f} minutes")
    print(f"Best Model Path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M4FT on Visuelle 2.0')
    
    # Path & Data
    parser.add_argument("--dataset_path", type=str, default='../visuelle2/')
    parser.add_argument("--ckpt_dir", type=str, default="ckpt_Gate_v3/") # [수정] 폴더 변경 권장
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Task Mode 
    parser.add_argument("--demand", type=int, default=1, 
                        help="1=Demand Task(No history), 0=SO-fore(With history)")
    parser.add_argument("--output_len", type=int, default=12) # Demand=12, SO-fore=1 or 10

    # Model Hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=32) 
    parser.add_argument('--hidden_dim', type=int, default=64)    
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    
    # Modality Flags
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0) 

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu_num", type=int, default=0)
    
    # Logging
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run", type=str, default="")

    args = parser.parse_args()
    
    # Safety Check
    if args.demand:
        args.output_len = 12

    run(args)