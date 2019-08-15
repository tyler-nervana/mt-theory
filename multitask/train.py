#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from data import make_teacher_classification
from tensorboardX import SummaryWriter


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    sample = next(data_loader_iter)
    try:
        writer.add_graph(model, sample["audio"])
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def train_models(model1, model2, optT, opt1, opt2, dl1, dl2, val_dl1, val_dl2, num_epochs=2000, multitask_freq=0, 
                 loss_fun1=nn.CrossEntropyLoss(), loss_fun2=nn.CrossEntropyLoss(), device="cpu", tb_writer=None,
                 w1=1.0, w2=1.0, update_every=True, store_gradients=False, single2=False, single_sample=False):
    multitask = (multitask_freq > 0)
    if single_sample:
        multitask_freq = 1
    iter1 = iter(dl1)
    iter2 = iter(dl2)

    train1_losses = []
    train2_losses = []
    test1_losses = []
    test2_losses = []
    test1_acc = []
    test2_acc = []
    for epoch in range(num_epochs):
        avg1_loss, count1 = 0, 0
        avg2_loss, count2 = 0, 0
        grads = {}
        model1.train()
        model2.train()
        for ii in range(len(dl1)):
            optT.zero_grad()
            if not single2:
                try:
                    x, y = next(iter1)
                except StopIteration:
                    #print("Stop iter hit at epoch {}, iter {}".format(epoch, ii))
                    iter1 = iter(dl1)
                    x, y = next(iter1)
                x, y = x.to(device), y.to(device)
                opt1.zero_grad()

                out = model1(x)
                loss1 = loss_fun1(out, y)
                avg1_loss += loss1.item()
                loss1 = w1 * loss1
                loss1.backward()
                opt1.step()
                count1 += 1
                if store_gradients:
                    for nm, param in mlp1.named_parameters():
                        if nm not in grads:
                            grads[nm] = 0
                        grads[nm] += param.grad.norm().item()
                if update_every:
                    optT.step()
                    optT.zero_grad()

            # Add multitask gradients
            if multitask or single2:
                opt2.zero_grad()                
                for mtiter in range(multitask_freq):
                    try:
                        new_x, y = next(iter2)
                    except StopIteration:
                        iter2 = iter(dl2)
                        new_x, y = next(iter2)
                    if not single_sample:
                        x = new_x
                    x, y = x.to(device), y.to(device)
                    out = model2(x)
                    loss2 = loss_fun2(out, y)
                    avg2_loss += loss2.item()                
                    loss2 = w2 * loss2
                    loss2.backward()
                    count2 += 1
                    if store_gradients:
                        for nm, param in mlp2.named_parameters():
                            if nm not in grads:
                                grads[nm] = 0
                            grads[nm] += param.grad.norm().item()
                    if update_every:
                        opt2.step()
                        optT.step()
                        opt2.zero_grad()
                        optT.zero_grad()
                if not update_every:
                    opt2.step()

            if not update_every:
                optT.step()

        if not single2:
            train1_losses.append(avg1_loss / count1)
        if multitask or single2:
            train2_losses.append(avg2_loss / count2)

        if not single2:
            avg_loss = 0
            correct = 0
            model1.eval()
            with torch.no_grad():
                for ii, (x, y) in enumerate(val_dl1):
                    x, y = x.to(device), y.to(device)
                    out = model1(x)
                    loss = loss_fun1(out, y)
                    avg_loss += loss.item()
                    correct += torch.eq(torch.max(out, dim=1)[1], y).sum().item() / len(y)
            test1_losses.append(avg_loss / (ii + 1))
            test1_acc.append(correct / (ii + 1))
            if (epoch % 10) == 0:
                print("Epoch {}: Train - {:.4f}, Test - {:.4f}, Acc - {:.3f}".format(epoch, train1_losses[-1], test1_losses[-1], test1_acc[-1]))
        
        if multitask or single2:
            avg_loss = 0
            correct = 0
            model2.eval()
            with torch.no_grad():
                for ii, (x, y) in enumerate(val_dl2):
                    x, y = x.to(device), y.to(device)
                    out = model2(x)
                    loss = loss_fun2(out, y)
                    avg_loss += loss.item()
                    correct += torch.eq(torch.max(out, dim=1)[1], y).sum().item() / len(y)
            test2_losses.append(avg_loss / (ii + 1))
            test2_acc.append(correct / (ii + 1))
            if (epoch % 10) == 0:
                print("Epoch {}: Train - {:.4f}, Test - {:.4f}, Acc - {:.3f}".format(epoch, train2_losses[-1], test2_losses[-1], test2_acc[-1]))

        if tb_writer is not None:
            if not single2:
                tb_writer.add_scalar("training-1/loss", train1_losses[-1], epoch)
                tb_writer.add_scalar("validation-1/loss", test1_losses[-1], epoch)
                tb_writer.add_scalar("validation-1/acc", test1_acc[-1], epoch)
            if multitask or single2:
                tb_writer.add_scalar("training-2/loss", train2_losses[-1], epoch)
                tb_writer.add_scalar("validation-2/loss", test2_losses[-1], epoch)
                tb_writer.add_scalar("validation-2/acc", test2_acc[-1], epoch)
            if store_gradients:
                for nm, val in grads.items():
                    if nm.startswith("trunk"):
                        l = count1 + count2
                    elif nm.startswith("output1"):
                        l = count1
                    elif nm.startswith("output2"):
                        l = count2
                    else:
                        raise RuntimeError("Encountered unexpected parameter: {}".format(nm))
                    tb_writer.add_scalar("grad-norm/{}".format(nm.replace(".", "-")),
                                         val / l,
                                         epoch)

    return (train1_losses, test1_losses, test1_acc), (train2_losses, test2_losses, test2_acc)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("Run toy multitask training")
    parser.add_argument("--nsamples", type=int, required=True, metavar="N",
                        help="Number of samples to generate per task dataset")
    parser.add_argument("--ntrain", type=int, metavar="N",
                        help="Number of training samples to use per task dataset. Default nsamples//2")
    parser.add_argument("--ntrain2", type=int, metavar="N",
                        help="Number of training samples to use for task 2. Default ntrain")
    parser.add_argument("--ntest", type=int, metavar="N",
                        help="Number of testing samples to use per task dataset. Default nsamples//2")
    parser.add_argument("--nfeatures", type=int, default=20, metavar="N",
                        help="Number of features in the dataset")
    parser.add_argument("--flip1", type=float, default=0, metavar="p",
                        help="Probability of flipping a class label")
    parser.add_argument("--flip2", type=float, default=0, metavar="p",
                        help="Probability of flipping a class label")    
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize second dataset to learn with only noise")
    parser.add_argument("--data_seed", type=int, default=0, metavar="k",
                        help="Seed for dataset RNG")
    parser.add_argument("--model_seed", type=int, default=0, metavar="k",
                        help="Seed for model RNG")
    parser.add_argument("--nunits", type=int, default=64, metavar="N",
                        help="Number of units per layer in MLP")
    parser.add_argument("--nlayers", type=int, default=2, metavar="N",
                        help="Number of layers in the MLP trunk")
    parser.add_argument("--save_dir", metavar="PATH", required=True,
                        help="Where to save model checkpoints and tensorboard outputs")
    parser.add_argument("--single", action="store_true",
                        help="Only train the first task")
    parser.add_argument("--nepochs", type=int, default=2000,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for all optimizers")
    parser.add_argument("--random_train", action="store_true",
                        help="Randomize training subset")
    parser.add_argument("--train_seed", type=int, default=0,
                        help="Seed for random training subset")
    parser.add_argument("--random_test", action="store_true",
                        help="Randomize testing subset")
    parser.add_argument("--test_seed", type=int, default=0,
                        help="Seed for random testing subset")
    parser.add_argument("--scale1", type=float, default=1,
                        help="Scale of task 1 SV")
    parser.add_argument("--scale2", type=float, default=1,
                        help="Scale of task 2 SV")
    parser.add_argument("--w1", type=float, default=1,
                        help="Weight for task 1 loss")
    parser.add_argument("--w2", type=float, default=1,
                        help="Weight for task 2 loss")
    parser.add_argument("--linear", action="store_true",
                        help="Use a linear network instead of ReLUs")
    parser.add_argument("--combine_grads", action="store_true",
                        help="Combine gradients from multiple tasks before updating")
    parser.add_argument("--copy_w", action="store_true",
                        help="Copy weights from task1 head to task2 head")
    parser.add_argument("--tie_w", action="store_true",
                        help="Tie weights from task1 head to task2 head")
    parser.add_argument("--use_task1_data", action="store_true",
                        help="Use task1 data for task2")
    parser.add_argument("--store_gradients", action="store_true",
                        help="Store parameter gradient norms")
    parser.add_argument("--no_dl_shuffle", action="store_true",
                        help="Do not shuffle samples in training dataloaders")
    parser.add_argument("--head1_lr", type=float,
                        help="lr for head1")
    parser.add_argument("--checkpoint", help="Path to checkpoint file to load")
    parser.add_argument("--single2", action="store_true",
                        help="Train only on task 2")
    parser.add_argument("--rank", type=int, default=1,
                        help="Rank of teacher networks")
    parser.add_argument("--relatedness", type=float, default=0,
                        help="Relatedness between two teacher networks")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of classes for each teacher network")
    parser.add_argument("--no_student_bias", dest="student_bias", action="store_false",
                        help="Add a bias to the student")
    parser.add_argument("--clean_validation", dest="noise_validation", action="store_false",
                        help="No noise added to validation data")
    parser.add_argument("--single_sample", action="store_true",
                        help="Use a single sample for all tasks")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size")
    parser.add_argument("--exp_values", action="store_true",
                        help="Use exponentially decaying singular values")
    
    args = parser.parse_args()
    
    if args.ntrain is None:
        args.ntrain = args.nsamples // 2

    if args.ntrain2 is None:
        args.ntrain2 = args.ntrain

    if args.ntest is None:
        args.ntest = args.nsamples // 2

    if args.head1_lr is None:
        args.head1_lr = args.lr

    args.dl_shuffle = (not args.no_dl_shuffle)

    # Write out args to a file
    print("Saving arguments to file")
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(vars(args), os.path.join(args.save_dir, "args.pt"))

    (xs1, ys1), (xs2, ys2), teacher1, teacher2 = make_teacher_classification(args.nsamples,
                                                            n_features=args.nfeatures,
                                                            rank=args.rank,
                                                            relatedness=args.relatedness,
                                                            flip_y1=args.flip1, flip_y2=args.flip2,
                                                            shuffle=True,
                                                            random_state=args.data_seed,
                                                            scale1=args.scale1,
                                                            scale2=args.scale2,
                                                            randomize=args.randomize,
                                                            n_classes=args.n_classes,
                                                            noise_validation=args.noise_validation,
                                                            exp_values=args.exp_values,
                                                            single_sample=args.single_sample)

    task1 = TensorDataset(torch.tensor(xs1).float(), torch.tensor(ys1).long())
    task2 = TensorDataset(torch.tensor(xs2).float(), torch.tensor(ys2).long())

    order = np.arange(0, len(task1))
    train_order = order[:len(task1)//2]
    test_order = order[len(task1)//2:]
    if args.random_train:
        generator = np.random.RandomState(seed=args.train_seed)        
        train_order = generator.permutation(train_order)
    if args.random_test:
        generator = np.random.RandomState(seed=args.test_seed)        
        test_order = generator.permutation(test_order)
        
    train_dl1 = DataLoader(Subset(task1, train_order[:args.ntrain]), batch_size=args.batch_size, drop_last=False, shuffle=args.dl_shuffle)
    test_dl1 = DataLoader(Subset(task1, test_order[:args.ntest]), batch_size=args.batch_size, drop_last=False, shuffle=False)
    print("Training task 1 on {} samples".format(len(train_dl1.dataset)))

    order = np.arange(0, len(task2))
    train_order = order[:len(task2)//2]
    test_order = order[len(task2)//2:]
    if args.random_train:
        generator = np.random.RandomState(seed=args.train_seed)        
        train_order = generator.permutation(train_order)
    if args.random_test:
        generator = np.random.RandomState(seed=args.test_seed)        
        test_order = generator.permutation(test_order)
    if args.use_task1_data:
        train_dl2 = DataLoader(Subset(task1, train_order[args.ntrain:(args.ntrain + args.ntrain2)]), batch_size=args.batch_size, drop_last=False,
                               shuffle=args.dl_shuffle)
        test_dl2 = DataLoader(Subset(task1, test_order[:args.ntest]), batch_size=args.batch_size, drop_last=False, shuffle=False)
    else:
        train_dl2 = DataLoader(Subset(task2, train_order[:args.ntrain2]), batch_size=args.batch_size, drop_last=False, shuffle=args.dl_shuffle)
        test_dl2 = DataLoader(Subset(task2, test_order[:args.ntest]), batch_size=args.batch_size, drop_last=False, shuffle=False)


    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = list()
    inputs = args.nfeatures
    for ii in range(args.nlayers):
        layers.append(nn.Linear(inputs, args.nunits, bias=args.student_bias))
        if not args.linear:
            layers.append(nn.ReLU())
        inputs = args.nunits

    trunk = nn.Sequential(*layers)
    output1 = nn.Linear(args.nunits, args.n_classes, bias=args.student_bias)
    output2 = nn.Linear(args.nunits, args.n_classes, bias=args.student_bias)
    for net in (trunk, output1, output2):
        for module in net.modules():
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.orthogonal_(module.weight)
    if args.tie_w:
        output2.weight = output1.weight
        output2.bias = output1.bias
    elif args.copy_w:  # Copy weights from output1 to output2
        output2.weight.data.copy_(output1.weight.data)
        output2.bias.data.copy_(output1.bias.data)

    # load checkpoint
    if args.checkpoint is not None:
        print("Loading weights from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        trunk.load_state_dict(checkpoint["trunk"])
        output1.load_state_dict(checkpoint["output1"])
        output2.load_state_dict(checkpoint["output2"])
        
    mlp1 = nn.Sequential(OrderedDict([("trunk", trunk), ("output1", output1)])).to(device)
    mlp2 = nn.Sequential(OrderedDict([("trunk", trunk), ("output2", output2)])).to(device)

    writer = create_summary_writer(mlp1, train_dl1, args.save_dir)

    optT = torch.optim.SGD(trunk.parameters(), lr=args.lr)
    opt1 = torch.optim.SGD(output1.parameters(), lr=args.head1_lr)
    opt2 = torch.optim.SGD(output2.parameters(), lr=args.lr)
    out = train_models(mlp1, mlp2, optT, opt1, opt2, train_dl1,
                       train_dl2, test_dl1, test_dl2,
                       num_epochs=args.nepochs,
                       multitask_freq=0 if args.single else (args.ntrain2 // args.ntrain),
                       device=device, tb_writer=writer,
                       w1=args.w1, w2=args.w2,
                       update_every=(not args.combine_grads),
                       store_gradients=args.store_gradients,
                       single2=args.single2,
                       single_sample=args.single_sample)

    torch.save({"trunk": trunk.state_dict(),
                "output1": output1.state_dict(),
                "output2": output2.state_dict()}, os.path.join(args.save_dir, "model.pt"))

    # Close tensorboard writer
    writer.close()
    
