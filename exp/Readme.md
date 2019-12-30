### 10038
```bash
nohup python run_classification_10038.py \
--data_file ../../data/pctr/10038/p_20191226/all.txt \
--header_file ../../data/pctr/10038/p_20191226.txt \
--label_col clk --embed_dim 32 --batch_size 64 2>&1 > log.txt &
```