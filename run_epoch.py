import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from utils.util import get_class

from models import *
from utils.loss import MCBCELoss
from mo import CBM
import copy


def run_epoch_without_deferral(args, model, data, optimizer, epoch, desc, device, loss_weight=None, train=False,  warm=False, inference_with_sampling=False, stage='joint'):
    if train:
        model.train()
        if isinstance(model, (ProbCBM)):
            if warm and hasattr(model, 'cnn_module'):
                for p in model.cnn_module.parameters():
                    p.requires_grad = False
            elif hasattr(model, 'cnn_module'):
                for p in model.cnn_module.parameters():
                    p.requires_grad = True
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset) * 2, args.num_labels).cpu()
    all_certainties = torch.zeros(len(data.dataset) * 2, args.num_concepts).cpu()
    all_cls_certainties = torch.zeros(len(data.dataset) * 2).cpu()
    all_targets = torch.zeros(len(data.dataset) * 2, args.num_labels).cpu()

    batch_idx = 0
    end_idx = 0
    loss_tot_dict = {'total': 0}

    # Set criterion for class and concept
    criterion_class = getattr(args, 'criterion_class', 'ce')
    if criterion_class == 'ce':
        criterion_class = nn.CrossEntropyLoss()
    else:
        raise ValueError('Got criterion_class', criterion_class)
    
    criterion_concept = getattr(args, 'criterion_concept', 'bce')
    if criterion_concept == 'bce':
        criterion_concept = nn.BCEWithLogitsLoss()
    elif criterion_concept == 'bce_prob':
        criterion_concept = nn.BCELoss()
    elif criterion_concept == 'MCBCELoss':
        in_criterion = nn.BCELoss(reduction='none')
        criterion_concept = get_class(criterion_concept, 'utils.loss')(criterion=in_criterion, reduction='mean', vib_beta=args.vib_beta, \
            group2concept=args.group2concept)
    else:
        raise ValueError('Got criterion_concept', criterion_concept)

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images = batch['image'].float().to(device)
        target_class = batch['class_label'][:, 0].long().to(device)
        target_concept = batch['concept_label'].float().to(device)

        if train:
            preds_dict, losses_dict = model(images, target_concept=target_concept, target_class=target_class, T=args.n_samples_train, stage=stage)
        else:
            with torch.no_grad():
                preds_dict, losses_dict = model(images, target_concept=target_concept, target_class=target_class, inference_with_sampling=inference_with_sampling, T=args.n_samples_inference)

        B = images.shape[0]
        class_label_onehot, concept_labels, labels, concept_uncertainty, class_uncertainty = None, None, None, None, None
        if args.pred_class:
            class_labels = batch['class_label'].float()
            class_label_onehot = torch.zeros(class_labels.size(0), args.num_classes)
            class_label_onehot.scatter_(1, class_labels.long(), 1)
            labels = class_label_onehot

        concept_labels = batch['concept_label'].float()
        if args.pred_concept:
            if labels is not None:
                labels = torch.cat((concept_labels, labels), 1)
            else:
                labels = concept_labels
        assert (labels is not None)

        loss, pred = 0, None
        loss_iter_dict = {}
        if args.pred_concept:
            if isinstance(criterion_concept, MCBCELoss):
                pred_concept = preds_dict['pred_concept_prob']
                loss_concept, concept_loss_dict = criterion_concept(\
                    probs=preds_dict['pred_concept_prob'],
                    image_mean=preds_dict['pred_mean'], image_logsigma=preds_dict['pred_logsigma'],
                    concept_labels=target_concept, negative_scale=preds_dict['negative_scale'], shift=preds_dict['shift'])
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']
                for k, v in concept_loss_dict.items():
                    if k != 'loss':
                        loss_iter_dict['pcme_' + k] = v
            elif isinstance(criterion_concept, (nn.BCELoss)):
                pred_concept = preds_dict['pred_concept_prob']
                loss_concept = criterion_concept(pred_concept, target_concept)
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']
            else:
                pred_concept = preds_dict['pred_concept_logit']
                loss_concept = criterion_concept(pred_concept, target_concept)
                pred_concept = torch.sigmoid(pred_concept)
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']

            if stage != 'class':
                loss += loss_concept * loss_weight['concept']
            pred = pred_concept
            loss_iter_dict['concept'] = loss_concept

        if args.pred_class:
            if 'pred_class_logit' in preds_dict.keys():
                pred_class = preds_dict['pred_class_logit']
                loss_class = criterion_class(pred_class, target_class)
                pred_class = F.softmax(pred_class, dim=-1)
            else:
                assert 'pred_class_prob' in preds_dict.keys()
                pred_class = preds_dict['pred_class_prob']
                loss_class = F.nll_loss(pred_class.log(), target_class, reduction='mean')
            loss_iter_dict['class'] = loss_class

            if stage != 'concept':
                loss += loss_class * loss_weight['class']
            pred = pred_class if pred is None else torch.cat((pred_concept, pred_class), dim=1)

            if 'pred_class_uncertainty' in preds_dict.keys():
                class_uncertainty = preds_dict['pred_class_uncertainty']

        for k, v in losses_dict.items():
            loss_iter_dict[k] = v
            if k in loss_weight.keys() and loss_weight[k] != 0:
                loss += v * loss_weight[k]
        loss_out = loss

        for k, v in loss_iter_dict.items():
            if v != v:
                print(k, v)

        if train:
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        ## Updates ##
        loss_tot_dict['total'] += loss_out.item()
        for k, v in loss_iter_dict.items():
            if k not in loss_tot_dict.keys():
                try:
                    loss_tot_dict[k] = v.item()
                except:
                    loss_tot_dict[k] = v
            else:
                try:
                    loss_tot_dict[k] += v.item()
                except:
                    loss_tot_dict[k] += v
        start_idx, end_idx = end_idx, end_idx + B

        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0), -1)

        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        if concept_uncertainty is not None:
            all_certainties[start_idx:end_idx] = concept_uncertainty.data.cpu()
        if class_uncertainty is not None:
            all_cls_certainties[start_idx:end_idx] = class_uncertainty.data.cpu()
        batch_idx += 1

    for k, v in loss_tot_dict.items():
        loss_tot_dict[k] = v / batch_idx


    return all_predictions[:end_idx], all_targets[:end_idx], all_certainties[:end_idx], all_cls_certainties[:end_idx], loss_tot_dict


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''

    batch_size = outputs.size()[0]  # batch_size
    rc = [n_classes-1] * batch_size
    outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
        outputs[range(batch_size), labels])  
    return torch.sum(outputs) / batch_size


def run_epoch(args, model, data, optimizer, epoch, desc, device, loss_weight=None, train=False,  warm=False, inference_with_sampling=False, stage='joint', expert_fn=None):



    if expert_fn is None:
        def f(x, y):
            if y < 100:
                return y
            else:
                if torch.rand(1) < 0.3:
                    return y
                else:
                    return 99
                
        expert_fn = f


    if train:
        model.train()
        if isinstance(model, (ProbCBM)):
            if warm and hasattr(model, 'cnn_module'):
                for p in model.cnn_module.parameters():
                    p.requires_grad = False
            elif hasattr(model, 'cnn_module'):
                for p in model.cnn_module.parameters():
                    p.requires_grad = True
        optimizer.zero_grad()
    else:
        model.eval()

    alpha = args.alpha

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset) * 2, args.num_labels).cpu()
    all_certainties = torch.zeros(len(data.dataset) * 2, args.num_concepts).cpu()
    all_cls_certainties = torch.zeros(len(data.dataset) * 2).cpu()
    all_targets = torch.zeros(len(data.dataset) * 2, args.num_labels).cpu()

    batch_idx = 0
    end_idx = 0
    loss_tot_dict = {'total': 0}

    # Set criterion for class and concept
    criterion_class = getattr(args, 'criterion_class', 'ce')
    if criterion_class == 'ce':
        criterion_class = reject_CrossEntropyLoss
    else:
        raise ValueError('Got criterion_class', criterion_class)
    

    criterion_concept = getattr(args, 'criterion_concept', 'bce')
    if criterion_concept == 'bce':
        criterion_concept = nn.BCEWithLogitsLoss()
    elif criterion_concept == 'bce_prob':
        criterion_concept = nn.BCELoss()
    elif criterion_concept == 'MCBCELoss':
        in_criterion = nn.BCELoss(reduction='none')
        criterion_concept = get_class(criterion_concept, 'utils.loss')(criterion=in_criterion, reduction='mean', vib_beta=args.vib_beta, \
            group2concept=args.group2concept)
    else:
        raise ValueError('Got criterion_concept', criterion_concept)

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images = batch['image'].float().to(device)
        target_class = batch['class_label'][:, 0].long().to(device)
        target_concept = batch['concept_label'].float().to(device)

        if train:
            preds_dict, losses_dict = model(images, target_concept=target_concept, target_class=target_class, T=args.n_samples_train, stage=stage)
        else:
            with torch.no_grad():
                preds_dict, losses_dict = model(images, target_concept=target_concept, target_class=target_class, inference_with_sampling=inference_with_sampling, T=args.n_samples_inference)

        B = images.shape[0]
        class_label_onehot, concept_labels, labels, concept_uncertainty, class_uncertainty = None, None, None, None, None
        if args.pred_class:
            class_labels = batch['class_label'].float()
            class_label_onehot = torch.zeros(class_labels.size(0), args.num_classes)
            class_label_onehot.scatter_(1, class_labels.long(), 1)
            labels = class_label_onehot

        concept_labels = batch['concept_label'].float()
        if args.pred_concept:
            if labels is not None:
                labels = torch.cat((concept_labels, labels), 1)
            else:
                labels = concept_labels
        assert (labels is not None)

        loss, pred = 0, None
        loss_iter_dict = {}
        if args.pred_concept:
            if isinstance(criterion_concept, MCBCELoss):
                pred_concept = preds_dict['pred_concept_prob']
                loss_concept, concept_loss_dict = criterion_concept(\
                    probs=preds_dict['pred_concept_prob'],
                    image_mean=preds_dict['pred_mean'], image_logsigma=preds_dict['pred_logsigma'],
                    concept_labels=target_concept, negative_scale=preds_dict['negative_scale'], shift=preds_dict['shift'])
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']
                for k, v in concept_loss_dict.items():
                    if k != 'loss':
                        loss_iter_dict['pcme_' + k] = v
            elif isinstance(criterion_concept, (nn.BCELoss)):
                pred_concept = preds_dict['pred_concept_prob']
                loss_concept = criterion_concept(pred_concept, target_concept)
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']
            else:
                pred_concept = preds_dict['pred_concept_logit']
                loss_concept = criterion_concept(pred_concept, target_concept)
                pred_concept = torch.sigmoid(pred_concept)
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']

            if stage != 'class':
                loss += loss_concept * loss_weight['concept']
            pred = pred_concept
            loss_iter_dict['concept'] = loss_concept

        # expert_pred = copy.deepcopy(target_class)
        if True:
            if 'pred_class_logit' in preds_dict.keys():
                pred_class = preds_dict['pred_class_logit']

            else:
                assert 'pred_class_prob' in preds_dict.keys()
                pred_class = preds_dict['pred_class_prob']
                pred_class = pred_class.log()

            pred_class = F.softmax(pred_class, dim=-1)

            m = []
            m2= torch.zeros(B)
            for j in range(B):
                expert_pred = expert_fn(images[j], target_class[j])
                if expert_pred == target_class[j]:
                    m.append(1)
                    m2[j] = alpha
                else:
                    m.append(0)
                    m2[j] = 1
            m = torch.tensor(m).to(device)
            m2 = torch.tensor(m2).to(device)
            pred_class.to(device)
            target_class.to(device)
            loss_class = criterion_class(pred_class, m, target_class, m2, args.num_classes)
            loss_iter_dict['class'] = loss_class

            if stage != 'concept':
                loss += loss_class * loss_weight['class']
            pred = pred_class if pred is None else torch.cat((pred_concept, pred_class), dim=1)

            if 'pred_class_uncertainty' in preds_dict.keys():
                class_uncertainty = preds_dict['pred_class_uncertainty']

        for k, v in losses_dict.items():
            loss_iter_dict[k] = v
            if k in loss_weight.keys() and loss_weight[k] != 0:
                loss += v * loss_weight[k]
        loss_out = loss

        for k, v in loss_iter_dict.items():
            if v != v:
                print(k, v)

        if train:
            print("Training")
            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()

        ## Updates ##
        loss_tot_dict['total'] += loss_out.item()
        for k, v in loss_iter_dict.items():
            if k not in loss_tot_dict.keys():
                try:
                    loss_tot_dict[k] = v.item()
                except:
                    loss_tot_dict[k] = v
            else:
                try:
                    loss_tot_dict[k] += v.item()
                except:
                    loss_tot_dict[k] += v
        start_idx, end_idx = end_idx, end_idx + B

        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0), -1)

        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        if concept_uncertainty is not None:
            all_certainties[start_idx:end_idx] = concept_uncertainty.data.cpu()
        if class_uncertainty is not None:
            all_cls_certainties[start_idx:end_idx] = class_uncertainty.data.cpu()
        batch_idx += 1

    # run_defferal_eval(args, model, data, device, desc, expert_fn=expert_fn)

    for k, v in loss_tot_dict.items():
        loss_tot_dict[k] = v / batch_idx

    return all_predictions[:end_idx], all_targets[:end_idx], all_certainties[:end_idx], all_cls_certainties[:end_idx], loss_tot_dict

def run_defferal_eval(args, model, data, device, desc, expert_fn=None):
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0

    if expert_fn is None:
        def f(x, y):
            if y < 100:
                return y
            else:
                if torch.rand(1) < 0.3:
                    return y
                else:
                    return 99
                
        expert_fn = f

    model.eval()

    with torch.no_grad():
        for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):
            images = batch['image'].float().to(device)
            target_class = batch['class_label'][:, 0].long().to(device)
            target_concept = batch['concept_label'].float().to(device)

            preds_dict, _ = model(images, target_concept=target_concept, target_class=target_class, T=1, stage='joint')

            B = images.shape[0]
            if 'pred_class_logit' in preds_dict.keys():
                pred_class = preds_dict['pred_class_logit']
            else:
                assert 'pred_class_prob' in preds_dict.keys()
                pred_class = preds_dict['pred_class_prob']
                pred_class = torch.log(pred_class)
            pred_class = F.softmax(pred_class, dim=-1)

            for i in range(B):
                expert_pred = expert_fn(images[i], target_class[i])
                prediction = torch.argmax(pred_class[i])
                r = (prediction == 200)

                class_pred = torch.argmax(pred_class[i][:-1])

                alone_correct += prediction == target_class[i]

                if r:
                    exp += expert_pred == target_class[i]
                    correct_sys += expert_pred == target_class[i]
                    exp_total += 1
                else:
                    correct += (class_pred == target_class[i]).item()
                    correct_sys += (class_pred == target_class[i]).item()
                    total += 1

                real_total += 1

    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    
    print(to_print)
    return correct, correct_sys, exp, exp_total, total, real_total, alone_correct





def run_epoch_cbm(args, model, data, optimizer, epoch, desc, device, loss_weight=None, train=False,  warm=False, inference_with_sampling=False, stage='joint', expert_fn=None):

    if expert_fn is None:
        def f(x, y):
            if y < 100:
                return y
            else:
                if torch.rand(1) < 0.3:
                    return y
                else:
                    return 99
                
        expert_fn = f

    if train:
        model.train()

    batch_idx = 0
    end_idx = 0
    loss_tot_dict = {'total': 0}

    criterion_class = getattr(args, 'criterion_class', 'ce')

    if criterion_class == 'ce':
        criterion_class = reject_CrossEntropyLoss
    else:
        raise ValueError('Got criterion_class', criterion_class)
    
    losses = []
    class_losses = []
    concept_losses = []

    alpha = args.alpha
    
    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):
        optimizer.zero_grad()
        images = batch['image'].float().to(device)
        target_class = batch['class_label'][:, 0].long().to(device)
        target_concept = batch['concept_label'].float().to(device)

        B = images.shape[0]
        

        num_concepts = args.num_concepts
    
        loss_iter_dict = {}

        concepts, preds = model(images)
        preds = F.softmax(preds, dim=-1)

        expert_pred = expert_fn(images, target_class)

        concept_loss = 0
        for i in range(num_concepts):
            concept_loss += torch.nn.CrossEntropyLoss()(concepts[i].squeeze(), target_concept[:, i].float().squeeze())

        m = []
        m2= torch.zeros(B)
        for j in range(B):
            expert_pred = expert_fn(images[j], target_class[j])
            if expert_pred == target_class[j]:
                m.append(1)
                m2[j] = alpha
            else:
                m.append(0)
                m2[j] = 1

        m = torch.tensor(m).to(device)
        m2 = torch.tensor(m2).to(device)

        class_loss = reject_CrossEntropyLoss(preds, m, target_class, m2, args.num_classes)


        loss = class_loss + concept_loss
        losses.append(loss)
        concept_losses.append(concept_loss)
        class_losses.append(class_loss)

        loss_out = loss

        loss_out.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss_out.item()} || Class Loss: {class_loss.item()} || Concept Loss: {concept_loss.item()}")

        del images, target_class, target_concept, preds

    # Average loss
    loss = sum(losses) / len(losses)
    class_loss = sum(class_losses) / len(class_losses)
    concept_loss = sum(concept_losses) / len(concept_losses)

    print(f"Epoch {epoch} Loss: {loss.item()} || Epoch Class Loss: {class_loss.item()} || Epoch Concept Loss: {concept_loss.item()}")

    return loss, class_loss, concept_loss
        


def run_defferal_eval_cbm(args, model, data, device, desc, expert_fn=None):
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0

    if expert_fn is None:
        def f(x, y):
            if y < 100:
                return y
            else:
                if torch.rand(1) < 0.3:
                    return y
                else:
                    return 99
                
        expert_fn = f

    model.eval()

    with torch.no_grad():
        for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):
            images = batch['image'].float().to(device)
            target_class = batch['class_label'][:, 0].long().to(device)
            target_concept = batch['concept_label'].float().to(device)

            concepts, preds = model(images)

            preds = F.softmax(preds, dim=-1)

            B = images.shape[0]
            pred_class = torch.argmax(preds, dim=1)

            for i in range(B):
                prediction = pred_class[i]
                r = pred_class[i] == 200

                class_pred = torch.argmax(preds[i][:-1])

                alone_correct += prediction == target_class[i]

                expert_pred = expert_fn(images[i], target_class[i])

                if r:
                    exp += expert_pred == target_class[i]
                    correct_sys += expert_pred == target_class[i]
                    exp_total += 1
                else:
                    correct += (class_pred == target_class[i]).item()
                    correct_sys += (class_pred == target_class[i]).item()
                    total += 1

                real_total += 1

    cov = str(total) + str(" out of") + str(real_total)

    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    
    print(to_print)
