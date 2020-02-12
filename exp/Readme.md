### 10038

#### embedding dimension
tensorflow推荐是 vocab_size**0.25，通常在3到300之间
https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html


#### 分割训练集验证集测试集
```bash
python split_train_val.py "../../data/pctr/10038/all/p_20200106/part*" \
 0.1 0.1 50 5 ../../data/pctr/10038/all/p_20200106/
```

#### 获取每一列特征的维度
```bash
python get_embed_dims.py "../../data/pctr/10038/p_20191226/part*" \
../../data/pctr/10038/p_20191226.txt var_len_cols1,var_len_cols2 \
../../data/pctr/10038/1226_header_dims.txt
```

#### v1
```bash
nohup python run_classification_10038.py \
--data_file ../../data/pctr/10038/p_20191226/all.txt \
--header_file ../../data/pctr/10038/p_20191226.txt \
--label_col clk --embed_dim 32 --batch_size 64 --use_cuda 2>&1 > log.txt &
```
#### v2
```bash
nohup python run_classification_10038_v2.py \
--data_file "../../data/pctr/10038/p_20191226/train_*" \
--val_file ../../data/pctr/10038/p_20191226/val \
--test_file ../../data/pctr/10038/p_20191226/test \
--col_dim_file ../../data/pctr/10038/1226_header_dims.txt \
--header_file ../../data/pctr/10038/p_20191226.txt \
--label_col clk --embed_dim 32 --batch_size 128 --use_cuda 2>&1 > log_adam.txt &
```
#### v3
```bash
nohup python run_classification_10038_v3.py \
--data_file "../../data/pctr/10038/all/p_20200106/train_*" \
--val_file "../../data/pctr/10038/all/p_20200106/val*" \
--test_file "../../data/pctr/10038/all/p_20200106/test*" \
--col_dim_file ../../data/pctr/10038/all/0106_header_dims.txt \
--header_file ../../data/pctr/10038/all/p_20200106.txt \
--label_col clk --embed_dim 32 --batch_size 256 --epoch_num 2 \
--query_varlen_feats wordIds --use_cuda 2>&1 > log_v3_2.txt &
```

#### inference获取广告侧特征
```bash
python dssm_inference.py --key_col id \
--header_file ../../data/pctr/10038/adside/p_20200106.txt \
--data_file "../../data/pctr/10038/adside/p_20200106/part*" \
--model ../../data/pctr/10038/all/best_0106.model \
--col_dim_file ../../data/pctr/10038/all/0106_header_dims.txt \
--query_varlen_feats wordIds --embed_dim 32 --use_cuda \
--out_path ../../data/pctr/10038/all/id_feat.txt
```
