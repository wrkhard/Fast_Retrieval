

import numpy as np
import argparse
import subprocess
import os
import pickle
from functools import partial
import matplotlib.pyplot as plt

import torch
from torch import nn
import pytorch_lightning as pl
import lightning_uq_box
from lightning_uq_box.models import ConditionalGuidedLinearModel
from lightning_uq_box.uq_methods import CARDRegression, DeterministicRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning import Trainer


from OCODataLoaders import RetrievalDataModule, RetrievalDataSet
from trainers import RetrievalTrainer, SimDiffTrainer
from models import LinearMLP, NonlinearMLP, Conv1DNN







def parser():
    parser = argparse.ArgumentParser(description='Train a retrieval model')
    parser.add_argument('--model_type', type=str, default='CondDiff', help='Model to train')
    parser.add_argument('--data_dir', type=str, default='/data/MLIA_active_data/data_FAST_RETRIEVAL/Data/', help='Directory containing data')
    parser.add_argument('--train_files', type=str, default=None, help='Training data file')
    parser.add_argument('--val_files', type=str, default=None, help='Validation data file')
    parser.add_argument('--test_files', type=str, default=None, help='Test data file')
    parser.add_argument('--save_dir', type=str, default='/data/MLIA_active_data/data_FAST_RETRIEVAL/Models/', help='Directory to save models')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--num_devices', type=int, default=0, help='Number of devices to train on')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    return parser.parse_args()








def main(args):
    
    model_type = args.model_type
    data_dir = args.data_dir
    train_files = args.train_files
    val_files = args.val_files
    test_files = args.test_files
    save_dir = args.save_dir
    verbose = args.verbose

    assert model_type in ['CondDiff', 'Evidential', 'SNGP', 'DeepEnsemble', 'MLE', 'MCSample'], 'Model type not recognized. Should be one of CondDiff, Evidential, SNGP, DeepEnsemble, MLE, MCSample'

    if train_files is None:
        train_list = [
            # 'L2DiaND_XCO2_2015_downsampled_eof_removed_aligned.p',
            # 'L2DiaND_XCO2_2016_downsampled_eof_removed_aligned.p',
            'L2DiaND_XCO2_2017_downsampled_eof_removed_aligned.p',
            'L2DiaND_XCO2_2018_downsampled_eof_removed_aligned.p',
            'L2DiaND_XCO2_2019_downsampled_eof_removed_aligned.p',
            'L2DiaND_XCO2_2020_downsampled_eof_removed_aligned.p',
            ]
        test_list = [
            'L2DiaND_XCO2_2021_downsampled_eof_removed_aligned.p',
            ]
    
        train_files = []
        test_files = []
        for file in train_list:
            train_files.append(os.path.join(data_dir, file))
        for file in test_list:
            test_files.append(os.path.join(data_dir, file))
        val_files = test_files

    # load data # TODO : Make sure val is sampled from train in the DataLoader
    dm = RetrievalDataModule(train_files, test_files, batch_size=32, val_split=0.1, num_workers=0, normalize_file='retrieval_normalize_stats.pkl')
    dm.setup('fit')

    INPUT_DIM = dm.get_input_shape()
    OUTPUT_DIM = dm.get_target_shape()
    INPUT_DIM = INPUT_DIM[1]
    OUTPUT_DIM = OUTPUT_DIM[1]

    # get number of gpus available
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    if len(available_gpus) > 0:
        DEVICE = 'cuda'
        NUM_DEVICES = args.num_devices
    else:
        DEVICE = 'cpu'
        NUM_DEVICES = 0



    if model_type == 'CondDiff':
        M = 1
        epochs = 10
        lr = 1e-6
        n_z_samples = 10
        n_steps = 100
        beta_start = 0.00001
        beta_end = 0.02
        cat_x = True  # condition on input x through concatenation
        cat_y_pred = True  # condition on y_0_hat

        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM

        n_hidden_backbone = [256, 256, 256]
        n_hidden_guidance = [512, 512]

        beta_schedule = "linear"

        early_stopping_callback = EarlyStopping(monitor='train_loss', patience=5, mode='min', verbose=verbose)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=1, save_last=True, verbose=verbose)

        preds_mean = []
        preds_epi = []
        preds_uct = []

        save_dir = save_dir + 'Diffusion/'
        for m in range(M):
            print('Training ensemble member : ' + str(m))
            network = NonlinearMLP(n_inputs=x_dim, n_hidden=n_hidden_backbone, n_outputs=y_dim, activation=nn.LeakyReLU())

            cond_mean_model = DeterministicRegression(
                model=network, optimizer=partial(torch.optim.Adam, lr=1e-5), loss_fn=nn.MSELoss(), 
            )
            trainer = Trainer(
                max_epochs=100,  # number of epochs we want to train
                devices=[0],
                accelerator="gpu",
                enable_checkpointing=False,
                enable_progress_bar=True,
            )
            trainer.fit(cond_mean_model, dm)
            # save the model
            torch.save(network.state_dict(), os.path.join(save_dir, f'pre_conditioner_{m}.pt'))

            guidance_model = ConditionalGuidedLinearModel(
                n_steps=n_steps,
                x_dim=x_dim,
                y_dim=y_dim,
                n_hidden=n_hidden_guidance,
                cat_x=cat_x,
                cat_y_pred=cat_y_pred,
                activation_fn=nn.LeakyReLU(),
            )
            card_model = CARDRegression(
                cond_mean_model=cond_mean_model.model,
                guidance_model=guidance_model,
                guidance_optim=partial(torch.optim.Adam, lr=1e-5),
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                n_steps=n_steps,
                n_z_samples=n_z_samples,
            )
            diff_trainer = Trainer(
                max_epochs=200,  # number of epochs we want to train

            )
            diff_trainer.fit(card_model, dm)
            # save the model
            torch.save(guidance_model.state_dict(), os.path.join(save_dir, f'card_model_{m}.pt'))

            print('Training complete')

            # predict on test set
            card_model = card_model.to("cpu")
            preds = card_model.predict_step(X_test)
            preds_mean.append(preds["pred"])
            preds_epi.append(preds["pred"])
            preds_uct.append(preds["pred_uct"])

        preds_mean = torch.stack(preds_mean, dim=0).mean(dim=0)
        preds_epi = torch.stack(preds_epi, dim=0).std(dim=0)
        preds_uct = torch.stack(preds_uct, dim=0).mean(dim=0)
        preds_uct = preds_uct + preds_epi

        # calculate the RMSE after unnormalizing the prediction
        Y_test = Y_test.squeeze()
        # open the normalize.pkl file and gt y_std and y_mean
        with open('normalize.pkl', 'rb') as file:
            norm_params = pickle.load(file)
        Y_test_unnormed = Y_test * norm_params['targets_std'] + norm_params['targets_mean']
        preds_mean = preds_mean * norm_params['targets_std'] + norm_params['targets_mean']
        preds_uct = preds_uct * norm_params['targets_std']
        Y_test_unnormed = Y_test_unnormed * 1e6
        preds_mean = preds_mean * 1e6
        # fig, ax = plt.subplots(1, figsize=(6, 6))
        # ax.scatter(Y_test_unnormed, preds_mean, label="XCO2 TCCON v. XCO2 NN")
        # ax.set_xlabel("XCO2 True")
        # ax.set_ylabel("XCO2 ML")
        # ax.set_xlim(390, 420)
        # ax.set_ylim(390, 420)
        # add the rmse to text
        rmse = np.sqrt(np.mean((Y_test_unnormed.cpu().numpy() - preds_mean.cpu().numpy())**2))
        # ax.text(0.1, 0.9, f"RMSE: {rmse:.2f} ppm", transform=ax.transAxes)
        # plt.legend()
        # plt.savefig(os.path.join(save_dir, "diffusion_ensemble_preds_V_true_2021.png"), dpi=300)
        print(f"RMSE: {rmse:.2f} ppm")

        # save the predictions to pickle
        pred_pickle = {
            "preds_mean": preds_mean,
            "preds_uct": preds_uct,
            "rmse_true": rmse,
            "rmse_tccon": "TBD",
        }
        with open(os.path.join(save_dir, "diffusion_ensemble_preds_2021.pkl"), "wb") as f:
            pickle.dump(pred_pickle, f)
        











            




    # TODO : Add Evidential, SNGP, DeepEnsemble, MLE, MCSample
    if model_type == 'Evidential':
        epochs = 1000
        batch_size = 32
        lr = 1e-6
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_hidden = [128, 128]
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=verbose)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=verbose)

    if model_type == 'SNGP':
        epochs = 1000
        batch_size = 32
        lr = 1e-6
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_hidden = [128, 128]
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=verbose)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=verbose)

    if model_type == 'DeepEnsemble':
        M = 30
        epochs = 1000
        batch_size = 32
        lr = 1e-6
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_hidden = [256, 256, 256, ]
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=verbose)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
    if model_type == 'MLE':
        epochs = 1000
        batch_size = 32
        lr = 1e-6
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_hidden = [256, 256, 256,]
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=verbose)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=verbose)

    if model_type == 'MCSample':
        epochs = 1000
        batch_size = 32
        lr = 1e-6
        n_samples = 1000
        beta_start = 0.00001
        beta_end = 0.02
        cat_x = True  # condition on input x through concatenation
        cat_y_pred = True  # condition on y_0_hat

        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM

        n_hidden = [128, 128]

        beta_schedule = "linear"

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=verbose)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=verbose)






if __name__ == '__main__':
    args = parser()
    main(args)











