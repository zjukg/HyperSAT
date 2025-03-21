### Code for paper *Structure-Aware Transformer for Hyper-Relational Knowledge Graph Completion*


For Wikipeople, to train and evalute on this dataset using default hyperparametes, please run:

```
python -u ./src/run.py --dataset "wikipeople" --device "0" --vocab_size 35005 --vocab_file "./data/wikipeople/vocab.txt" \
                       --train_file "./data/wikipeople/train+valid.json" --test_file "./data/wikipeople/test.json" \
                       --ground_truth_file "./data/wikipeople/all.json" --num_workers 1 --num_relations 178 \
                       --num_entities 34825 --max_seq_len 13 --max_arity 7 --hidden_dim 256 local_layers 2 \
                       --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 \
                       --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 \
                       --hyperedge_dropout 0.99 --epoch 300 --warmup_proportion 0.1 --neighbor_num 5
```

For WD50K, to train and evalute on this dataset using default hyperparametes, please run:

```
python -u ./src/run.py --dataset "wd50k" --device "0" --vocab_size 47688 --vocab_file "./data/wd50k/vocab.txt" \
                       --train_file "./data/wd50k/train+valid.json" --test_file "./data/wd50k/test.json" \
                       --ground_truth_file "./data/wd50k/all.json" --num_workers 1 --num_relations 531 \
                       --num_entities 47155 --max_seq_len 15 --max_arity 8 --hidden_dim 512 --local_layers 2 \
                       --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 512 \
                       --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 \
                       --hyperedge_dropout 0.0 --epoch 300 --warmup_proportion 0.1 --neighbor_num 5 --residual_w 0.5
```

For FB15K237, to train and evalute on this dataset using default hyperparametes, please run:
```
python -u ./src/run.py --dataset "fb15k237" --device "1" --vocab_size 14780 --vocab_file "./data/fb15k237/vocab.txt" \
                       --train_file "./data/fb15k237/train.json" --test_file "./data/fb15k237/test.json" \
                       --ground_truth_file "./data/fb15k237/all.json" --num_workers 1 --num_relations 237 \
                       --num_entities 14541 --max_seq_len 3 --max_arity 2 --hidden_dim 256 --local_layers 2\
                       --local_dropout 0.5 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 \
                       --lr 0.0005 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hid_times 4\
                       --hyperedge_dropout 0.0 --epoch 400 --warmup_proportion 0.1 --neighbor_num 0 --residual_w 0.2
```
For WN18RR, to train and evalute on this dataset using default hyperparametes, please run:
```
python -u ./src/run.py --dataset "wn18rr" --device "2" --vocab_size 40956 --vocab_file "./data/wn18rr/vocab.txt" \
                       --train_file "./data/wn18rr/train.json" --test_file "./data/wn18rr/test.json" \
                       --ground_truth_file "./data/wn18rr/all.json" --num_workers 1 --num_relations 11 \
                       --num_entities 40943 --max_seq_len 3 --max_arity 2 --hidden_dim 256 --local_layers 8\
                       --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 \
                       --lr 0.001 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hid_times 4\
                       --hyperedge_dropout 0.0 --epoch 300 --warmup_proportion 0.1 --neighbor_num 0 --residual_w 0.2
```

For wikidata12k, to train and evalute on this dataset using default hyperparametes, please run:
```
python -u ./src/run.py --dataset "wikidata12k" --device "2" --vocab_size 13201 --vocab_file "./data/wikidata12k/vocab.txt" \
                       --train_file "./data/wikidata12k/train.json" --test_file "./data/wikidata12k/test.json" \
                       --ground_truth_file "./data/wikidata12k/all.json" --num_workers 1 --num_relations 26 \
                       --num_entities 12554 --max_seq_len 7 --max_arity 4 --hidden_dim 256 --local_layers 2\
                       --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 \
                       --lr 1e-3 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 \
                       --hyperedge_dropout 0.0 --epoch 400 --warmup_proportion 0.1 --neighbor_num 1 --residual_w 0.5