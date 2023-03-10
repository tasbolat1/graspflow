from os import stat
import torch
from networks.models import GraspEvaluator
from networks import losses
from datasets import GraspDatasetWithTight as GraspDataset
from torch.utils.data import DataLoader
#from tqdm.auto import tqdm
from networks.utils import save_model, add_weight_decay
import time
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter

import stats

# python train_evaluator.py --data_dir_pcs data/pcs --batch_size 400 --lr 0.0001 --device_ids 0 1 2


# sampler run
parser = argparse.ArgumentParser("Train Grasp Evaluator model.")
parser.add_argument("--tag", type=str, help="Frequency to test model.", default='evaluator')
parser.add_argument("--epochs", type=int, help="Number of epochs.", default=1000)
# parser.add_argument("--data_dir_grasps", type=str, help="Path to data.", required=True)
parser.add_argument("--data_dir_pcs", type=str, help="Path to data.", required=True)
parser.add_argument("--lr", type=float, help="Learning rate.", default=0.0001)
parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer.", default=0.0)
parser.add_argument("--batch_size", type=int, help="Batch Size.", default=128)
parser.add_argument("--num_workers", type=int, help="Number of workers for dataloader.", default=20)
parser.add_argument("--save_freq", type=int, help="Frequency to save model.", default=1)
parser.add_argument("--test_freq", type=int, help="Frequency to test model.", default=1)
parser.add_argument("--continue_train", action='store_true', help="Continue to train: checkpoint_dir must be indicated")
parser.add_argument("--checkpoint_info", type=str, help="Checkpoint info as: name_epoch", default=None)
parser.add_argument("--device_ids", nargs="+", type=int, help="Index of cuda devices. Pass -1 to set to cpu.", default=[0])
args = parser.parse_args()


# prepare device ids
if len(args.device_ids) > 1:
    device = torch.device(f'cuda:{args.device_ids[0]}')
else:
    if args.device_ids[0] == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device_ids[0]}')

# prepare model
model = GraspEvaluator()
weight_decay = args.weight_decay
if weight_decay:
    parameters = add_weight_decay(model, weight_decay)
    weight_decay = 0.
else:
    parameters = model.parameters()
optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay, lr=args.lr)

if args.continue_train:
    assert args.checkpoint_info != None, 'please indicate checkpoint info.'
    model_name = args.checkpoint_info.split('_')[0]
    init_epoch = int(args.checkpoint_info.split('_')[1])
    model_save_path = f'saved_models/{args.tag}/{model_name}/'
    print(f'Continue to train {model_save_path} starting from {init_epoch} epoch ...')
    model.load_state_dict(torch.load(f'{model_save_path}/{init_epoch}.pt'))
else:
    model_name= int( time.time()*100 )
    model_save_path = f'saved_models/{args.tag}/{model_name}/'
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    init_epoch = 0

model.to(device)
if len(args.device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total learnable params: {total_params}.')

# prepare to save to tensorboard
writer = SummaryWriter(model_save_path)

# prepare data
#train_dataset = GraspDataset(path_to_grasps = args.data_dir_grasps, path_to_pc=args.data_dir_pcs, split='train', augment=True)
train_dataset = GraspDataset(path_to_grasps = 'data/grasps_lower/preprocessed2',
                             path_to_grasps_tight='data/grasps_tight_lower/preprocessed2',
                             path_to_pc=args.data_dir_pcs, split='train', augment=True, mode=2)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#test_dataset = GraspDataset(path_to_grasps = args.data_dir_grasps, path_to_pc=args.data_dir_pcs, split='test', augment=False)
test_dataset1 = GraspDataset(path_to_grasps = 'data/grasps_lower/preprocessed2',
                             path_to_grasps_tight='data/grasps_tight_lower/preprocessed2',
                             path_to_pc=args.data_dir_pcs, split='test', augment=False, mode=0)
test_dataloader1 = DataLoader(test_dataset1, batch_size=args.batch_size+200, shuffle=False, num_workers=args.num_workers)

test_dataset2 = GraspDataset(path_to_grasps = 'data/grasps_lower/preprocessed2',
                             path_to_grasps_tight='data/grasps_tight_lower/preprocessed2',
                             path_to_pc=args.data_dir_pcs, split='test', augment=False, mode=1)
test_dataloader2 = DataLoader(test_dataset2, batch_size=args.batch_size+200, shuffle=False, num_workers=args.num_workers)


criteria_eval = torch.nn.BCEWithLogitsLoss(reduction='mean')
criteria_trans = torch.nn.MSELoss()

len_trainloader = len(train_dataloader)
len_testloader1 = len(test_dataloader1)
len_testloader2 = len(test_dataloader2)

print(f'len_trainloader: {len_trainloader} for {len(train_dataset)}')
print(f'len_testloader1: {len_testloader1} for {len(test_dataset1)}')
print(f'len_testloader2: {len_testloader2} for {len(test_dataset2)}')

C1 = 0.850 # eval
C2 = 0.149 # quat
C3 = 0.001 # trans

def train(epoch):

    loss_eval_metr = stats.AverageMeter('train/loss_eval', writer)
    loss_quat_metr = stats.AverageMeter('train/loss_quat', writer)
    loss_trans_metr = stats.AverageMeter('train/loss_trans', writer)
    loss_total_metr = stats.AverageMeter('train/loss', writer)

    accuracy = stats.BinaryAccuracyWithCat('train', writer)

    # train
    model.train()
    for k, (quat, trans, pcs, labels, cats, _) in enumerate(train_dataloader):
        
        # move to device
        quat, trans, pcs, labels = quat.to(device), trans.to(device), pcs.to(device), labels.to(device)
        
        # normalize pcs
        pc_mean = pcs.mean(dim=1).unsqueeze(1)
        pcs = pcs - pc_mean
        trans = trans-pc_mean.squeeze(1)
        
        # forward pass
        eval_out, quat_out, trans_out = model(quat, trans, pcs)
        eval_out = eval_out.squeeze(-1)
        #print(out.shape)
        
        # compute loss
        labels = labels.squeeze(-1)

        loss_eval = criteria_eval(eval_out, labels)
        loss_quat = losses.quat_chordal_squared_loss(quat_out, quat)
        loss_trans = criteria_trans(trans_out, trans)

        loss = C1*loss_eval + C2*loss_quat + C3*loss_trans
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_eval_metr.update(loss_eval.item(), quat.shape[0])
        loss_quat_metr.update(loss_quat.item(), quat.shape[0])
        loss_trans_metr.update(loss_trans.item(), quat.shape[0])
        loss_total_metr.update(loss.item(), quat.shape[0])
        accuracy.update(eval_out.squeeze().detach().cpu().numpy(), labels.detach().cpu().numpy(), cats)

        if k % 1000 == 0:
            print( loss_eval_metr.summary(epoch, some_txt=f'{k}/{len_trainloader}') )
            print( loss_quat_metr.summary(epoch, some_txt=f'{k}/{len_trainloader}') )
            print( loss_trans_metr.summary(epoch, some_txt=f'{k}/{len_trainloader}') )
            print( loss_total_metr.summary(epoch, some_txt=f'{k}/{len_trainloader}') )
        
    # print( train_loss.summary(epoch) )
    print( accuracy.summary(epoch) )


def test2(epoch):

    loss_eval_metr = stats.AverageMeter('test/loss_eval', writer)
    loss_quat_metr = stats.AverageMeter('test/loss_quat', writer)
    loss_trans_metr = stats.AverageMeter('test/loss_trans', writer)
    loss_total_metr = stats.AverageMeter('test/loss', writer)

    accuracy = stats.BinaryAccuracyWithCat('test', writer)

    # train
    model.eval()
    with torch.no_grad():
        for k, (quat, trans, pcs, labels, cats, _) in enumerate(test_dataloader2):
            
            # move to device
            quat, trans, pcs, labels = quat.to(device), trans.to(device), pcs.to(device), labels.to(device)
            
            # normalize pcs
            pc_mean = pcs.mean(dim=1).unsqueeze(1)
            pcs = pcs - pc_mean
            trans = trans-pc_mean.squeeze(1)
            
            # forward pass
            eval_out, quat_out, trans_out = model(quat, trans, pcs)
            eval_out = eval_out.squeeze(-1)
            #print(out.shape)
            
            # compute loss
            labels = labels.squeeze(-1)

            loss_eval = criteria_eval(eval_out, labels)
            loss_quat = losses.quat_chordal_squared_loss(quat_out, quat)
            loss_trans = criteria_trans(trans_out, trans)

            loss = C1*loss_eval + C2*loss_quat + C3*loss_trans

            loss_eval_metr.update(loss_eval.item(), quat.shape[0])
            loss_quat_metr.update(loss_quat.item(), quat.shape[0])
            loss_trans_metr.update(loss_trans.item(), quat.shape[0])
            loss_total_metr.update(loss.item(), quat.shape[0])

            accuracy.update(eval_out.squeeze().detach().cpu().numpy(), labels.detach().cpu().numpy(), cats)
                
    print( loss_eval_metr.summary(epoch, some_txt=f'{k}/{len_testloader2}') )
    print( loss_quat_metr.summary(epoch, some_txt=f'{k}/{len_testloader2}') )
    print( loss_trans_metr.summary(epoch, some_txt=f'{k}/{len_testloader2}') )
    print( loss_total_metr.summary(epoch, some_txt=f'{k}/{len_testloader2}') )
    print( accuracy.summary(epoch) )


def test1(epoch):

    loss_eval_metr = stats.AverageMeter('test1/loss_eval', writer)
    loss_quat_metr = stats.AverageMeter('test1/loss_quat', writer)
    loss_trans_metr = stats.AverageMeter('test1/loss_trans', writer)
    loss_total_metr = stats.AverageMeter('test1/loss', writer)

    accuracy = stats.BinaryAccuracyWithCat('test1', writer)

    # train
    model.eval()
    with torch.no_grad():
        for k, (quat, trans, pcs, labels, cats, _) in enumerate(test_dataloader1):
            
            # move to device
            quat, trans, pcs, labels = quat.to(device), trans.to(device), pcs.to(device), labels.to(device)
            
            # normalize pcs
            pc_mean = pcs.mean(dim=1).unsqueeze(1)
            pcs = pcs - pc_mean
            trans = trans-pc_mean.squeeze(1)
            
            # forward pass
            eval_out, quat_out, trans_out = model(quat, trans, pcs)
            eval_out = eval_out.squeeze(-1)
            #print(out.shape)
            
            # compute loss
            labels = labels.squeeze(-1)

            loss_eval = criteria_eval(eval_out, labels)
            loss_quat = losses.quat_chordal_squared_loss(quat_out, quat)
            loss_trans = criteria_trans(trans_out, trans)

            loss = C1*loss_eval + C2*loss_quat + C3*loss_trans

            loss_eval_metr.update(loss_eval.item(), quat.shape[0])
            loss_quat_metr.update(loss_quat.item(), quat.shape[0])
            loss_trans_metr.update(loss_trans.item(), quat.shape[0])
            loss_total_metr.update(loss.item(), quat.shape[0])

            accuracy.update(eval_out.squeeze().detach().cpu().numpy(), labels.detach().cpu().numpy(), cats)
                
    print( loss_eval_metr.summary(epoch, some_txt=f'{k}/{len_testloader1}') )
    print( loss_quat_metr.summary(epoch, some_txt=f'{k}/{len_testloader1}') )
    print( loss_trans_metr.summary(epoch, some_txt=f'{k}/{len_testloader1}') )
    print( loss_total_metr.summary(epoch, some_txt=f'{k}/{len_testloader1}') )
    print( accuracy.summary(epoch) )

# # test inital values
test1(init_epoch)
test2(init_epoch)


for epoch in range(init_epoch+1,args.epochs+1):
    train(epoch)
    if epoch % args.test_freq == 0:
        test1(epoch)
        test2(epoch)
    
    # save model
    if epoch % args.save_freq == 0:
        save_model(model, path=model_save_path, epoch=epoch)
        
    print(f'Done with epoch {epoch} ...')

    if epoch == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00075
    elif epoch == 2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif epoch == 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00025
    
    elif epoch == 4:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001