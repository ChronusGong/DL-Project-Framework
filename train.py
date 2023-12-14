import time
import os

import numpy as np
import pandas as pd
import random
import torch
import argparse
import scipy.stats as st
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import explained_variance_score, max_error, median_absolute_error, mean_absolute_error, mean_squared_error, r2_score

from data_prepare.TrainingData import get_dataset
from data_prepare.label_split import scaler
from models.choose_model import choose_model

SEED_LIST = [random.randint(0,1000)for i in range(5)]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# dataset
parser.add_argument('--label_file', type=str, default='./block_income.csv', help='File name for storing label information')
parser.add_argument('--test_size', type=float, default=0.05, help='Set the size of the validation set and the test set')
parser.add_argument('--seed', type=int, default=SEED_LIST, nargs='+', help='Set random seeds')
parser.add_argument('--EMB_DIR', type=str, default='./data/street_view_emb_768', help='The path to the embedding folder')
# models
parser.add_argument('--models', type=str, default=['ResNet34', 'DRSN', 'EfficientNet', 'CNN'],
                    nargs='+', help='Model name')
# training
parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--num_epochs', type=int, default=10, help='# of epochs')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()


def get_writer(timestr, seed, model_name):
    time_path = os.path.join('runs', timestr)
    if not os.path.exists(time_path):
        os.mkdir(time_path)
    seed_path = os.path.join(time_path, str(seed))
    if not os.path.exists(seed_path):
        os.mkdir(seed_path)
    writer = SummaryWriter(seed_path + model_name)
    return writer


def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    loss_sum = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
    for step, (X, y) in pbar:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        pred = pred.view(pred.shape[0])
        loss = criterion(pred, y)

        if torch.isnan(loss):
            print("Nan Loss!")
            print(pred, y)
            raise ValueError("Nan Loss!")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum = loss_sum * (step / (step + 1)) + loss.item() / (step + 1)
        pbar.set_postfix({"loss": loss_sum, "lr": optimizer.state_dict()['param_groups'][0]['lr']})

    writer.add_scalar('Train/Loss', loss_sum / len(train_loader), epoch)
    writer.flush()


def valid(model, device, val_loader, criterion, epoch, writer):
    model.eval()
    loss_sum = 0
    current_val_loss = 0

    for step, (X, y) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validating..."):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            pred = model(X)
            pred = pred.view(pred.shape[0])
            loss = criterion(pred, y)
            if torch.isnan(loss):
                print("Nan Loss!")
                print(pred, y)
                raise ValueError("Nan Loss!")

            loss_sum = loss_sum * (step / (step+1)) + loss.item() / (step+1)
    writer.add_scalar('Test/Loss', loss_sum / len(val_loader), epoch)
    writer.flush()
    return loss_sum / len(val_loader)


def test(model, device, test_loader):
    model.eval()

    pred_list = []
    target_list = []

    for step, (X, y) in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing..."):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            pred = model(X)
            pred = pred.view(pred.shape[0])
            pred = pred.to(torch.float32)

            pred = pred.to('cpu')
            y = y.to('cpu')

            pred_list += list(pred)
            target_list += list(y)

    metrics = {}
    metrics["scaled_explained_variance"] = explained_variance_score(target_list, pred_list)
    metrics["scaled_mean_absolute_error"] = mean_absolute_error(target_list, pred_list)
    metrics["scaled_mean_squared_error"] = mean_squared_error(target_list, pred_list)
    metrics["scaled_r2"] = r2_score(target_list, pred_list)

    target_inversed = scaler.inverse_transform(np.array(target_list).reshape(-1, 1))
    pred_inversed = scaler.inverse_transform(np.array(pred_list).reshape(-1, 1))

    metrics["unscaled_max_error"] = max_error(target_inversed, pred_inversed)
    metrics["unscaled_mean_absolute_error"] = mean_absolute_error(target_inversed, pred_inversed)
    metrics["unscaled_median_absolute_error"] = median_absolute_error(target_inversed, pred_inversed)
    return metrics


def cal_avg_and_confidence(metrics_list):
    temp_dict = {}
    avg_dict = {}
    confidence_dict = {}
    for metrics in metrics_list:
        for k, v in metrics.items():
            if k not in temp_dict.keys():
                temp_dict[k] = []
            temp_dict[k].append(v)

    for k, v in temp_dict.items():
        avg_dict[k] = np.mean(v)
        confidence_dict[k] = st.t.interval(0.95, df=len(v)-1,
                                           loc=np.mean(v), scale=st.sem(v))
    return avg_dict, confidence_dict



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    criterion = nn.L1Loss()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    avg_dict = {}
    confidence_dict = {}
    for model_name in args.models:

        print(f"Training {model_name}...")

        time_path = os.path.join('model', timestr)
        if not os.path.exists(time_path):
            os.mkdir(time_path)
        model_path = os.path.join(time_path, model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        metrics_list = []
        for index, seed in enumerate(args.seed):
            print(f'Model {model_name} Seed {index + 1}: {seed}')
            writer = get_writer(timestr, seed, model_name)
            model = choose_model(model_name).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            train_dataSet, val_dataSet, test_dataSet = get_dataset(args.label_file, args.test_size, seed, args.EMB_DIR)

            train_loader = DataLoader(train_dataSet, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataSet, batch_size=args.batch_size)
            test_loader = DataLoader(test_dataSet, batch_size=args.batch_size)

            # train and validation
            best_val_loss = torch.inf
            early_stop_cnt = 0
            start = time.time()
            for i in range(args.num_epochs):
                train(model, device, train_loader, criterion, optimizer, i, writer)
                val_loss = valid(model, device, val_loader, criterion, i, writer)
                if val_loss < best_val_loss:
                    print("Better model found! Saving...")
                    torch.save(model, f"./model/{timestr}/{model_name}/best.bin")
                    best_val_loss = val_loss
                else:
                    early_stop_cnt += 1
                    print(f"Early stop cnt: {early_stop_cnt}")
                    if early_stop_cnt >= 2:
                        print(f"Early stop count threshold reached, stopping...")
                        break
                torch.save(model, f'./model/{timestr}/{model_name}/{i}.bin')
                writer.flush()
            time_all = time.time() - start
            print('Training complete in {:.0f}m {:.0f}s'.format(time_all // 60, time_all % 60))
            writer.close()

            # test
            metrics = test(model, device, test_loader)
            metrics_list.append(metrics)

        avg_dict[model_name], confidence_dict[model_name] = cal_avg_and_confidence(metrics_list)

    avg_df = pd.DataFrame(avg_dict)
    confidence_df = pd.DataFrame(confidence_dict)


    # scaled_df = pd.DataFrame(scaled_dict)
    # unscaled_df = pd.DataFrame(unscaled_dict)
    res_path = os.path.join('result', timestr)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_df.to_csv(f"./result/{timestr}/avg.scv")
    confidence_df.to_csv(f"./result/{timestr}/confidence.scv")


if __name__ == '__main__':
    main()






