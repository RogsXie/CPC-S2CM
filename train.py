import sys
import time
from scipy.optimize import linear_sum_assignment
sys.path.append('/')
from scipy import io
import os
import numpy as np
import torch
import argparse
from modules import dataset, network, loss, transform
from utils import yaml_config_hook, save_model, metric, initialization_utils
import torch
from Toolbox import Preprocessing
import csv
import os
import pandas as pd
torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, loss_op, train_loader, optimizer):
    model.train()
    loss_epoch = 0
    for step, ((x_1, x_2), y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list_1 = [x_i.to(DEVICE) for x_i in x_1]
        x_list_2 = [x_i.to(DEVICE) for x_i in x_2]
        y1, y2 = model(x_list_1, x_list_2)
        loss, loss_con, loss_clu = loss_op(y1, y2, model.clustering_head.cluster_centers)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t loss: "  f"{loss_.item():.6f}\t" f'CL:{loss_con.item():.6f}\t CLU: {loss_clu.item():.6f}')
        loss_epoch += loss_.item()
    return loss_epoch


def inference(test_loader, model, device, is_labeled_pixel):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(test_loader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            pred = model.forward_cluster(x_list)
        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    y_pred_vector = np.array(y_pred_vector)
    labels_vector = np.array(labels_vector)

    if is_labeled_pixel:
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(labels_vector, y_pred_vector)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y_true = labels_vector[indx_labeled]
        y_pred = y_pred_vector[indx_labeled]

        true_classes = np.unique(y_true)  #
        pred_classes = np.unique(y_pred)

        n_true = len(true_classes)
        n_pred = len(pred_classes)
        confusion_matrix = np.zeros((n_true, n_pred), dtype=np.int64)

        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = np.where(true_classes == true_label)[0][0]
            pred_idx = np.where(pred_classes == pred_label)[0][0]
            confusion_matrix[true_idx, pred_idx] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        label_map = {
            pred_classes[col_idx]: true_classes[row_idx]
            for row_idx, col_idx in zip(row_ind, col_ind)
        }

        y_pred_mapped = np.array([label_map.get(p, -1) for p in y_pred])

        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(y_true, y_pred_mapped)

    print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}'.format(acc, kappa, nmi, ari, pur))

    GT = io.loadmat(gt_path)
    gt = GT['GT']  # Trento
    # gt = GT['gt']   # MUUFL&Augsburg

    Preprocessing.Processor().show_class_map(y_pred_mapped, indx_labeled, gt)
    return acc, kappa, nmi, ari, pur, ca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    pretrain_path = args.model_path + '/pretrain'
    joint_train_path = args.model_path + '/joint-train'
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    if not os.path.exists(joint_train_path):
        os.makedirs(joint_train_path)
    initialization_utils.set_global_random_seed(seed=args.seed)

    root = args.dataset_root

    # prepare data
    if args.dataset == "Trento":
        im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Augsburg":
        im_1, im_2 = 'data_HS_LR', 'data_SAR_HR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "MUUFL":
        im_1, im_2 = 'HSI', 'LiDAR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    else:
        raise NotImplementedError
    gt_path = root + gt_ + '.mat'
    dataset_train = dataset.MultiModalDataset(gt_path, *img_path, patch_size=(args.image_size, args.image_size),
                                              transform=transform.Transforms(size=args.image_size),
                                              is_labeled=False)
    class_num = dataset_train.n_classes
    print('Processing %s ' % img_path[0])
    print(dataset_train.data_size, class_num)
    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        prefetch_factor=4
    )

    # # test loader
    dataset_test = dataset.MultiModalDataset(gt_path, *img_path,
                                             patch_size=(args.image_size, args.image_size),
                                             transform=None, is_labeled=args.is_labeled_pixel)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=512,
                                                   shuffle=False,
                                                   drop_last=False, num_workers=args.workers)

    model = network.Net(dataset_train.in_channels, class_num, args.dim_emebeding)

    model = model.to(DEVICE)

    # optimizer / loss
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if 'clustering_head' not in n],
         'lr': args.learning_rate},
        {"params": model.clustering_head.cluster_centers, 'lr': args.learning_rate * args.lr_scale}
    ]
    optimizer = torch.optim.Adam(grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # # ===== joint training ==========
    score_list = []
    each_class = []
    max_acc = 0
    best_ca = None
    best_metrics = None
    acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
    score_list.append([acc, kappa, nmi, ari, pur])
    print(f'initial accuracy: ACC={acc:.4f}')


    loss_op_joint = loss.JointLoss(args.batch_size,  # class_num,  #
                                   lambda_=args.contrastive_param,
                                   weight_clu=args.weight_clu_loss,
                                   regularization_coef=args.regularizer_coef, device=DEVICE)

    loss_history = []

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print('start fine-tuning ...')
    start_time = time.time()
    for epoch in range(1, args.joint_train_epoch + 1):
        loss_epoch = train(model, loss_op_joint, data_loader_train, optimizer)
        print(f"Epoch [{epoch}/{args.joint_train_epoch}]\t Loss: {loss_epoch / len(data_loader_train)}")
        if epoch % 1 == 0:
            acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
            score_list.append([acc, kappa, nmi, ari, pur])
            each_class.append([ca])
            if acc > max_acc:
                max_acc = acc
                best_ca = ca
                best_metrics = [acc, kappa, nmi, ari, pur]
                print('Better acc')
            # save_model(joint_train_path, model, optimizer, epoch)
        loss_history.append(loss_epoch / len(data_loader_train))
        lr_scheduler.step()
    running_time = time.time() - start_time
    print(f'fine tuning time: {running_time:.3f} s')
    save_model(joint_train_path, model, optimizer, args.joint_train_epoch)
    print(loss_history)
    print(score_list)
    print(each_class)

    output_dir = os.path.join("OUTPUT", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    # 准备数据
    data = []

    if best_ca is not None:
        for idx, ca_val in enumerate(best_ca):
            data.append({'Class': f'Class_{idx}', 'MCPC': f"{ca_val * 100:.2f}"})

    metric_names = ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity']
    for name, metric in zip(metric_names, best_metrics):
        data.append({'Class': name, 'MCPC': f"{metric * 100:.2f}"})

    data.append({'Class': 'Running Time', 'MCPC': f"{running_time:.2f}"})

    df = pd.DataFrame(data)

    csv_file = os.path.join(output_dir, 'best_results.csv')

    df.to_csv(csv_file, index=False)

    print(f"bestcsv: {csv_file}")
    print(df)

