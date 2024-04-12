import torch
import torch.nn as nn
import os, shutil

# from adamp import AdamP

from models import *
from config import DEBUG, get_configs
import utils.evaluate as evaluate
from utils.util import get_class, load_saved_model
from run_epoch import run_epoch, run_defferal_eval, run_epoch_cbm, run_defferal_eval_cbm
from mo import CBM
from torch.optim import SGD

args = get_configs()

print("\n=========================================")
print("Arguments")
for arg in vars(args):
    print(f'{arg}: \t{getattr(args, arg)}')
print("=========================================")
if 'wandb' in args.log_tool:
    import wandb

# Change how to set device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())
    print('')
else:
    device = torch.device('mps')
    print('Current device: cpu')

class2concept, group2concept = None, None
if args.dataset == 'cub':
    from dataloaders.cub312_datamodule import get_data
    loaders, class2concept, group2concept = get_data(args)
args.group2concept = group2concept

from torchvision.models.resnet import resnet18
model = CBM(resnet18(pretrained=True), 201, 313, 0.2, 0, 'relu')

# print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("# of params: ", pytorch_total_params)
model = model.to(device)

loss_weight = {'concept': args.loss_weight_concept, 'class': args.loss_weight_class}

if args.train_class_mode in ['joint', 'independent']:
    stages = ['joint']
elif args.train_class_mode == 'sequential':
    stages = ['concept', 'class']

epoch = 0
for stage in stages:
    # Set optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    start_epoch = 0
    val_best_acc = 0
    for stage_epoch in range(start_epoch + 1, args.epochs_class + 1 if stage == 'class' else args.epochs + 1):
        epoch += 1
        print('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            print('LR: {}'.format(param_group['lr']))

        run_epoch_cbm(args, model, loaders['train'], optimizer, epoch, stage, device=device, loss_weight=loss_weight)

        run_defferal_eval_cbm(args, model, loaders['val'], device, "Eval", None)



# if __name__ == '__main__':
#     ############## Log and Save ##############
#     all_preds, all_targs, all_certs, all_cls_certs, test_loss = run_epoch(args, model, loaders['test'], None, 0, 'Testing', device=device, loss_weight=loss_weight)
#     test_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_certs, all_cls_certs, test_loss['total'], 0)
#     print('Test Acc: ', test_metrics['class_acc'])
#     print('Test Loss: ', test_loss['total'])
