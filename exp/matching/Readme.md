
```bash
export DATA_DIR=../data/nmt/jdpair/20200814
python exp/matching/split_train_val.py $DATA_DIR/segquery.txt $DATA_DIR/segitem.txt \ 
 $DATA_DIR/pair.txt  0.01 0.01 $DATA_DIR/train/
```