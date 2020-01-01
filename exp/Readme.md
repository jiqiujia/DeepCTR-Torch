### 10038
```bash
nohup python run_classification_10038.py \
--data_file ../../data/pctr/10038/p_20191226/all.txt \
--header_file ../../data/pctr/10038/p_20191226.txt \
--label_col clk --embed_dim 32 --batch_size 64 --use_cuda 2>&1 > log.txt &
```

```bash
python get_embed_dims.py "../../data/pctr/10038/p_20191226/part*" \
../../data/pctr/10038/p_20191226.txt \
../../data/pctr/10038/1226_header_dims.txt
```

```bash
nohup python run_classification_10038_v2.py \
--data_file "../../data/pctr/10038/p_20191226/train_*" \
--val_file ../../data/pctr/10038/p_20191226/val \
--test_file ../../data/pctr/10038/p_20191226/test \
--col_dim_file ../../data/pctr/10038/1226_header_dims.txt \
--header_file ../../data/pctr/10038/p_20191226.txt \
--label_col clk --embed_dim 32 --batch_size 64 --use_cuda 2>&1 > log2.txt &
```