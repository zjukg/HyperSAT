import argparse
import logging
import torch
import torch.nn
import torch.optim
import os
import sys
import time
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from reader import *
from evaluation import *
from transformer import *
from graph import *

parser = argparse.ArgumentParser()

# Dataset information
parser.add_argument("--dataset", type=str, default="icews14") #"jf17k", 
parser.add_argument("--vocab_size", type=int, default=8090) #29148
parser.add_argument("--vocab_file", type=str, default="./data/icews14/vocab.txt") #"./data/jf17k/vocab.txt"
parser.add_argument("--train_file", type=str, default="./data/icews14/train.json") #"./data/jf17k/train.json"
parser.add_argument("--test_file", type=str, default="./data/icews14/test.json") #"./data/jf17k/test.json"
parser.add_argument("--ground_truth_file", type=str, default="./data/icews14/all.json") #"./data/jf17k/all.json"
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_relations", type=int, default=595) #501
parser.add_argument("--num_entities", type=int, default=595) #501
parser.add_argument("--max_seq_len", type=int, default=5) #11
parser.add_argument("--max_arity", type=int, default=3) #6

# Hyperparameter
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--global_layers", type=int,  default=12)
parser.add_argument("--global_dropout", type=float, default=0.2) # 0.2
parser.add_argument("--global_activation", type=str, default="elu") # relu, elu, gelu, tanh
parser.add_argument("--global_heads", type=int,  default=4) # 4
parser.add_argument("--local_layers", type=int,  default=2) # 12
parser.add_argument("--local_dropout", type=float, default=0.2) # 0.2
parser.add_argument("--local_heads", type=int, default=4) # 4
parser.add_argument("--decoder_activation", type=str, default="gelu") # relu, elu, gelu, tanh
parser.add_argument("--batch_size", type=int, default=2048) # 1024
parser.add_argument("--lr", type=float, default=5e-4) # 5e-4
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--entity_soft", type=float, default=0.9) # 0.9
parser.add_argument("--relation_soft", type=float, default=0.0) # 0.0
parser.add_argument("--hyperedge_dropout", type=float, default=0.0) # dropout rate of hyperedge learning
parser.add_argument("--device", type=str, default="2") # {0123}^n,1<=n<=4,the first cuda is used as master device and others are used for data parallel
parser.add_argument("--remove_mask", type=bool, default=False) # wheather to use extra mask
parser.add_argument("--residual_w", type=float, default=0.5) # 
parser.add_argument("--neighbor_num", type=int, default=0)
parser.add_argument("--positional", type=bool, default=True)
parser.add_argument("--fact_index", type=bool, default=True)
parser.add_argument("--kvpair", type=bool, default=True)
parser.add_argument("--comp", type=bool, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hid_times", type=int, default=2)

# others for training
parser.add_argument("--epoch", type=int, default=100) # 200
parser.add_argument("--warmup_proportion", type=float, default=0.1)

# Ablation experiment
parser.add_argument("--use_edge", type=bool, default=True)
parser.add_argument("--use_node", type=bool, default=True)

# directory position settings
parser.add_argument("--result_save_dir", type=str, default="results")
parser.add_argument("--ckpt_save_dir", type=str, default="ckpts")

args = parser.parse_args()
# args.num_entities = args.vocab_size - args.num_relations - 2
args.temp = False
if args.dataset in ['wikidata12k', 'yago11k']:
    args.temp = True
if not os.path.exists(args.result_save_dir):
    os.mkdir(args.result_save_dir)
if not os.path.exists(args.ckpt_save_dir):
    os.mkdir(args.ckpt_save_dir)
parentdir = os.path.join(args.result_save_dir,args.dataset)
if not os.path.exists(parentdir):
    os.mkdir(parentdir)
existing = sorted([int(x) for x in os.listdir(parentdir) if x.isdigit()], reverse=True)
if not existing:
        # If no subfolder exists
        dir_name = os.path.join(parentdir,'0')
        os.mkdir(dir_name)
        # dir_name.mkdir()
else:
        # If there are subfolders and we want to make a new dir
        dir_name = os.path.join(parentdir,str(existing[0] + 1))
        # dir_name = parentdir / str(existing[0] + 1)
        # dir_name.mkdir()
        os.mkdir(dir_name)
logging.basicConfig(
    format='%(asctime)s  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    filename=f'{dir_name}/train.log',
    filemode="a",
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
def main(limit=1e9):
    # prepare GPU or GPUs
    device = torch.device(f"cuda:{args.device[0]}")
    devices = []
    for i in range(len(args.device)):
        devices.append(torch.device(f"cuda:{args.device[i]}"))
    
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)

    vocabulary = Vocabulary(
        vocab_file=args.vocab_file,
        num_relations=args.num_relations,
        num_entities=args.num_entities)
    train_examples, _ , ent_related_neighbors = read_examples(args.train_file, args.max_arity, args.dataset, train=True)
    test_examples, _ = read_examples(args.test_file, args.max_arity, args.dataset, train=False)
    train_dataset = MultiDataset(vocabulary, train_examples, args.max_arity, args.max_seq_len, args.neighbor_num, ent_related_neighbors,train_mode=True)
    test_dataset = MultiDataset(vocabulary, test_examples, args.max_arity, args.max_seq_len, args.neighbor_num, ent_related_neighbors, train_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=1)
    max_train_steps = args.epoch * len(train_loader)
    # graph = build_graph(vocabulary, train_examples, args.hyperedge_dropout, device)
    
    if len(devices) > 1:
        model = torch.nn.DataParallel(Transformer(args.vocab_size, args.max_seq_len, args.neighbor_num, args.residual_w, args.positional, args.fact_index, args.comp, args.local_layers, args.hidden_dim,
                                                  args.local_heads, args.local_dropout, args.decoder_activation, args.use_edge, args.remove_mask, args.use_node, args.hid_times), device_ids=devices)
        model.to(device)
    else:
        model = Transformer(args.vocab_size, args.max_seq_len, args.neighbor_num, args.residual_w, args.positional, args.fact_index, args.comp, args.local_layers, args.hidden_dim,
                                                  args.local_heads, args.local_dropout, args.decoder_activation, args.use_edge, args.remove_mask, args.use_node, args.hid_times).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=max_train_steps, 
                            pct_start=args.warmup_proportion, anneal_strategy="linear", cycle_momentum=False)
    limit = min(args.epoch, limit)

    training_time = int(0)
    best_mrr = 0

    for epoch in range(limit):
        time_start_epoch = time.time()
        for item in tqdm(train_loader):
            model.train()
            item = (i.to(device) for i in item)
            input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type = item
            result = model(input_ids, input_mask, mask_position, mask_output, edge_labels)
            entities, relations = (query_type == 1), (query_type == -1)

            label_entity = mask_output[entities] * (args.entity_soft / (args.num_entities - 1))
            label_entity[torch.arange(label_entity.shape[0]), mask_label[entities]] = 1 - args.entity_soft
            label_relation = mask_output[relations] * (args.relation_soft / (args.num_relations - 1))
            label_relation[torch.arange(label_relation.shape[0]), mask_label[relations]] = 1 - args.relation_soft            
            loss1 = torch.nn.functional.cross_entropy(result[entities], label_entity, reduction='none')
            loss2 = torch.nn.functional.cross_entropy(result[relations], label_relation, reduction='none')
            loss = torch.cat((loss1, loss2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        time_end_epoch = time.time()
        training_time += round(time_end_epoch - time_start_epoch)
        hours, minutes, seconds = calculate_training_time(training_time)
        logger.info(f"epoch: {epoch}\tlr: {scheduler.get_last_lr()[0]:.6f}\ttrain time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        if epoch % 5 == 0 or epoch == limit - 1:
            eval_performance = predict(
                model=model,
                test_loader=test_loader,
                all_features=test_dataset.features,
                vocabulary=vocabulary,
                device=device)
            show_perforamance(eval_performance)
            Exp_name_params = [
                args.dataset,
                "nb_"+str(args.neighbor_num),
                "layer_"+str(args.local_layers),
                "dim_"+str(args.hidden_dim)
            ]
            cur_mrr = eval_performance['entity']['mrr']
            if(cur_mrr>best_mrr):
                torch.save(
                    model.state_dict(),
                    os.path.join(args.ckpt_save_dir, "_".join(Exp_name_params)+".ckpt"),
                    )
                best_mrr = cur_mrr
def calculate_training_time(training_time: int):
    minutes, seconds = divmod(training_time, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def predict(model, test_loader, all_features, vocabulary, device):
    eval_result_file = os.path.join(dir_name, "eval_result.json")

    gt_dict = generate_ground_truth(
        ground_truth_path=args.ground_truth_file,
        vocabulary=vocabulary,
        max_arity=args.max_arity,
        max_seq_length=args.max_seq_len,
        dataset=args.dataset)

    step = 0
    global_idx = 0
    ent_lst = []
    rel_lst = []
    _2_r_lst = []
    _2_ht_lst = []
    _n_r_lst = []
    _n_ht_lst = []
    _n_a_lst = []
    _n_v_lst = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(test_loader):
            item = (i.to(device) for i in item)
            input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type = item

            output = model(input_ids, input_mask, mask_position, mask_output, edge_labels)

            batch_results = output.cpu().numpy()
            ent_ranks, rel_ranks, _2_r_ranks, _2_ht_ranks, \
            _n_r_ranks, _n_ht_ranks, _n_a_ranks, _n_v_ranks = batch_evaluation(
                global_idx, batch_results, all_features, gt_dict, args.temp, args.num_relations, args.num_entities)
            ent_lst.extend(ent_ranks)
            rel_lst.extend(rel_ranks)
            _2_r_lst.extend(_2_r_ranks)
            _2_ht_lst.extend(_2_ht_ranks)
            _n_r_lst.extend(_n_r_ranks)
            _n_ht_lst.extend(_n_ht_ranks)
            _n_a_lst.extend(_n_a_ranks)
            _n_v_lst.extend(_n_v_ranks)
            step += 1
            global_idx += output.size(0)

    eval_result = compute_metrics(
        ent_lst=ent_lst,
        rel_lst=rel_lst,
        _2_r_lst=_2_r_lst,
        _2_ht_lst=_2_ht_lst,
        _n_r_lst=_n_r_lst,
        _n_ht_lst=_n_ht_lst,
        _n_a_lst=_n_a_lst,
        _n_v_lst=_n_v_lst,
        eval_result_file=eval_result_file
    )

    return eval_result

def show_perforamance(eval_performance):
    def pad(x):
        return x + (10 - len(x)) * ' '
    all_entity = f"{pad('ENTITY')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['entity']['mrr'],
        eval_performance['entity']['hits1'],
        eval_performance['entity']['hits3'],
        eval_performance['entity']['hits5'],
        eval_performance['entity']['hits10'])

    all_relation = f"{pad('RELATION')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['relation']['mrr'],
        eval_performance['relation']['hits1'],
        eval_performance['relation']['hits3'],
        eval_performance['relation']['hits5'],
        eval_performance['relation']['hits10'])

    all_ht = f"{pad('HEAD/TAIL')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['ht']['mrr'],
        eval_performance['ht']['hits1'],
        eval_performance['ht']['hits3'],
        eval_performance['ht']['hits5'],
        eval_performance['ht']['hits10'])

    all_r = f"{pad('PRIMARY_R')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['r']['mrr'],
        eval_performance['r']['hits1'],
        eval_performance['r']['hits3'],
        eval_performance['r']['hits5'],
        eval_performance['r']['hits10'])

    logger.info("\n-------- Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join([pad("TASK"), "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        all_ht, all_r, all_entity, all_relation))

def profile():
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.add_function(predict)
    lp.add_function(batch_evaluation)
    wrapper = lp(main)
    wrapper(1)
    lp.print_stats()

if __name__ == '__main__':
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')
    main()
    #profile()
