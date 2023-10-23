# # # 完全消融
# # 40 单模态数据
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type log --dataset original --data ../data/chunk_10  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type kpi --dataset original --data ../data/chunk_10  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type log --dataset yzh --data ../data/data3 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type kpi --dataset yzh --data ../data/data3 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type log --dataset zte --data ../data/zte2  --patience 10
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type kpi --dataset zte --data ../data/zte2  --patience 10

# # # 40 单模态 gan
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type log --dataset original --data ../data/chunk_10 --open_gan True  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type kpi --dataset original --data ../data/chunk_10 --open_gan True  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type log --dataset yzh --data ../data/data3 --open_gan True 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type kpi --dataset yzh --data ../data/data3 --open_gan True 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type log --dataset zte --data ../data/zte2 --open_gan True  --patience 10
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type kpi --dataset zte --data ../data/zte2 --open_gan True  --patience 10

# # # # 40 多模态融合 wo gan wo unmatch
# # # cross_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type cross_attn  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type cross_attn 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type cross_attn  --patience 10
# # # sep_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type sep_attn  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type sep_attn 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type sep_attn  --patience 10
# # # concat
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type concat  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type concat 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type concat  --patience 10

# # # 40 多模态融合 with gan wo unmatch
# # # cross_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type cross_attn --open_gan True  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type cross_attn --open_gan True 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type cross_attn --open_gan True  --patience 10
# # # sep_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type sep_attn --open_gan True  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type sep_attn --open_gan True 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type sep_attn --open_gan True  --patience 10
# # # concat
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type concat --open_gan True  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type concat --open_gan True 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type concat --open_gan True  --patience 10

# # # 40 多模态融合 with unmatch wo gan
# # # cross_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type cross_attn --open_unmatch_zoomout True   --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type cross_attn --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type cross_attn --open_unmatch_zoomout True   --patience 10
# # # sep_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type sep_attn --open_unmatch_zoomout True   --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type sep_attn --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type sep_attn --open_unmatch_zoomout True   --patience 10
# # # concat
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type concat --open_unmatch_zoomout True   --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type concat --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type concat --open_unmatch_zoomout True   --patience 10


# # # 40 多模态融合 with gan unmatch
# # # cross_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type cross_attn --open_gan True --open_unmatch_zoomout True    --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type cross_attn --open_gan True --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type cross_attn --open_gan True --open_unmatch_zoomout True   --patience 10
# # # sep_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type sep_attn --open_gan True --open_unmatch_zoomout True    --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type sep_attn --open_gan True --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type sep_attn --open_gan True --open_unmatch_zoomout True   --patience 10
# # # concat
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type concat --open_gan True --open_unmatch_zoomout True    --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type concat --open_gan True --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --fuse_type concat --open_gan True --open_unmatch_zoomout True   --patience 10

# # 极简消融
# # fuse gan 消融实验
# 40 wo gan wo unmatch
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2  --patience 10
# # 40 unmatch wo gan
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_unmatch_zoomout True --open_expand_anomaly_gap True --open_narrowing_modal_gap True
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_unmatch_zoomout True   --patience 10
# # 40 gan wo unmatch
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_gan True  --patience 10
# # 40 gan unmatch
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True    --open_expand_anomaly_gap True --open_narrowing_modal_gap True
/home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_gan True --open_unmatch_zoomout True   --patience 10


# 专项参数敏感
# for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_unmatch_zoomout True --unmatch_k $k  --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_unmatch_zoomout True --unmatch_k $k 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_unmatch_zoomout True --unmatch_k $k  --patience 10
# done

# 完全体参数敏感实验
# hidden_size (16,32,64,128,256,512)
# window_size (10,20,30,40,50,60,70,80,90,100)
# unmatch_k (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) (1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0) (2,4,6,8,10,12,14,16,18,20)

# for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# for k in 20
# do 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True --unmatch_k $k --open_expand_anomaly_gap True --open_narrowing_modal_gap True # 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True --unmatch_k $k # 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_gan True --open_unmatch_zoomout True --unmatch_k $k --patience 10 --window_size 40 # --hidden_size 64 # --unmatch_k 16
# done

# for w in 10 20 30 40 50 60 70 80 90 100
# do 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True  --window_size $w --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True  --window_size $w
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_gan True --open_unmatch_zoomout True  --window_size $w --patience 10
# done

# for h in 16 32 64 128 256 512
# do
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True  --hidden_size $h --open_expand_anomaly_gap True --open_narrowing_modal_gap True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True  --hidden_size $h
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset zte --data ../data/zte2 --open_gan True --open_unmatch_zoomout True  --hidden_size $h --patience 10
# done









# 消融实验
# loss trick
# --open_narrowing_modal_gap True
# --open_expand_anomaly_gap





# # gan 消融实验
# 多模态 gan wo unmatch
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout False --open_gan_sep False
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout False --open_gan_sep False
# # 多模态 gan with unmatch
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True --open_gan_sep False
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True --open_gan_sep False
# # 多模态 wo gan unmatch
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type multi_modal_self_attn --open_gan False --open_unmatch_zoomout False
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type multi_modal_self_attn --open_gan False --open_unmatch_zoomout False

# # 多模态 gan wo unmatch with sep_gan
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout False --open_gan_sep True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout False --open_gan_sep True
# # 多模态 gan with unmatch with sep_gan
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True --open_gan_sep True
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True --open_gan_sep True



# unmatch_k (1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0) (2,4,6,8,10,12,14,16,18,20)
# for ((i=1; i<=50; i++))  
# do  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True --unmatch_k $((i*2)) 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True --unmatch_k $((i*2)) 
# done
# for ((i=0; i<=5; i++))  
# do  
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --open_gan True --open_unmatch_zoomout True --unmatch_k $((-i*2)) 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --open_gan True --open_unmatch_zoomout True --unmatch_k $((-i*2)) 
# done

# # # 默认 multi_modal_self_attn 
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset original --data ../data/chunk_10 --fuse_type multi_modal_self_attn
# /home/hongyi/.conda/envs/hades/bin/python run.py --gpu_device 0 --data_type fuse --dataset yzh --data ../data/data3 --fuse_type multi_modal_self_attn
