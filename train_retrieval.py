

import numpy as np
import argparse
import subprocess
import os
import pickle
from functools import partial
import matplotlib.pyplot as plt

import torch
from torch import nn
import lightning as pl
import lightning_uq_box
from lightning_uq_box.models import ConditionalGuidedLinearModel, MLP
from lightning_uq_box.models.fc_resnet import FCResNet
from lightning_uq_box.uq_methods import CARDRegression, DeterministicRegression, DER, SNGPRegression, DeepEnsembleRegression, MVERegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning import Trainer


from OCODataLoaders import RetrievalDataModule, RetrievalDataSet
from trainers import RetrievalTrainer, SimDiffTrainer
from models import LinearMLP, NonlinearMLP, Conv1DNN

# TODO bring in timm lr schedulers.





def parser():
    def parse_hidden(arg):
        return list(map(int, arg.split(',')))
    # Global parser
    parser = argparse.ArgumentParser(description='Train a retrieval model')
    parser.add_argument('--model_type', type=str, default='CondDiff', help='Model to train')
    parser.add_argument('--data_dir', type=str, default='/data/MLIA_active_data/data_FAST_RETRIEVAL/Data/', help='Directory containing data')
    parser.add_argument('--train_files', type=str, default=None, help='Training data file')
    parser.add_argument('--test_files', type=str, default=None, help='Test data file')
    parser.add_argument('--save_dir', type=str, default='/data/MLIA_active_data/data_FAST_RETRIEVAL/Models/', help='Directory to save models')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--num_devices', type=int, default=1, help='Number of devices to train on')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    parser.add_argument('--batch_size', type=int, default=150, help='Batch size')

    # pca args
    parser.add_argument('--pca_bands', type=list, default=None, help='Bands to use for PCA')
    parser.add_argument('--pca_components', type=int, default=10, help='Number of PCA components')

    # Global lr and callbacks -- CondDiff uses different callbacks
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum delta for early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # Conditional Diffusion Hyperparameters
    # BEST CONFIG SO FAR: (torch) [wkeely@paralysis Code]$ python train_retrieval.py --num_devices 0 --model_type 'CondDiff' --beta_scheduler 'cosine_anneal' --n_steps 10 --conditional_mean_epochs 100 --guidance_model_epochs 150 --lr_backbone 1e-4 --lr_guidance 1e-6 --n_z_samples 100 --beta_start 0.0000001 --beta_end 0.00002 --cat_x --cat_y_pred --n_hidden_backbone 256,256,128 --n_hidden_guidance 128,128 --min_delta_cond 0.0 --patience_cond 10 --min_delta_guidance 0.0 --patience_guidance 5
    parser.add_argument('--conditional_mean_epochs', type=int, default=50, help='Number of epochs to train conditional mean model')
    parser.add_argument('--guidance_model_epochs', type=int, default=150, help='Number of epochs to train guidance model')
    parser.add_argument('--lr_backbone', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_guidance', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--n_z_samples', type=int, default=100, help='Number of samples for z')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of steps for diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0000001, help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.00002, help='Ending beta value')
    parser.add_argument('--cat_x', action='store_true', help='Condition on input x through concatenation')
    parser.add_argument('--cat_y_pred', action='store_true', help='Condition on y_0_hat')
    parser.add_argument('--n_hidden_backbone', type=parse_hidden,  default=[256,256,128], help='Number of hidden units in each layer')
    parser.add_argument('--n_hidden_guidance', type=parse_hidden,  default=[128,128], help='Number of hidden units in each layer')
    parser.add_argument('--beta_scheduler', type=str, default='cosine_anneal_schedule', help='Noise scheduler')
    # parser.add_argument('--M', type=int, default=1, help='Number of ensemble members')
    parser.add_argument('--min_delta_cond', type=float, default=0.0, help='Minimum delta for early stopping')
    parser.add_argument('--patience_cond', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--min_delta_guidance', type=float, default=0.0, help='Minimum delta for early stopping')
    parser.add_argument('--patience_guidance', type=int, default=5, help='Patience for early stopping')


    # Evidential Hyperparameters
    parser.add_argument('--evidential_epochs', type=int, default=250, help='Number of epochs to train evidential model')
    parser.add_argument('--n_hidden_evidential', type=parse_hidden, default=[256,256,128], help='Number of hidden units in each layer')



    # SNGP Hyperparameters
    parser.add_argument('--sngp_epochs', type=int, default=150, help='Number of epochs to train SNGP model')
    parser.add_argument('--n_features', type=int, default=256, help='Number of features in the feature extractor')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the feature extractor')

    # Deep Ensemble Hyperparameters
    parser.add_argument('--deep_ensemble_epochs', type=int, default=100, help='Number of epochs to train Deep Ensemble')
    parser.add_argument('--n_hidden_deep_ensemble', type=parse_hidden,default=[256,256,128], help='Number of hidden units in each layer')
    parser.add_argument('--M', type=int, default=1, help='Number of ensemble members')

    # MLE Hyperparameters
    parser.add_argument('--mle_epochs', type=int, default=100, help='Number of epochs to train MLE model')
    parser.add_argument('--n_hidden_mle', type=parse_hidden, default=[256,256,256], help='Number of hidden units in each layer')




    return parser.parse_args()





# TODO : Finalize check point saving and loading.
# TODO : Address RMSE nan issue that randomly appears.
# TODO : AGU Abstract Due July 31st. Make plots for Diffusion model retrieval results and KDE.


def main(args):
    
    model_type = args.model_type
    data_dir = args.data_dir
    train_files = args.train_files
    test_files = args.test_files
    save_dir = args.save_dir
    verbose = args.verbose

    assert model_type in ['CondDiff', 'Evidential', 'SNGP', 'DeepEnsemble', 'MLE', 'MCSample'], 'Model type not recognized. Should be one of CondDiff, Evidential, SNGP, DeepEnsemble, MLE, MCSample'
    # if args.model_type == 'CondDiff':
    #     args.batch_size = 50
    if train_files is None:
        train_list = [
            # 'L2DiaND_XCO2_2015_downsampled_eof_removed_aligned.p',
            # 'L2DiaND_XCO2_2016_downsampled_eof_removed_aligned.p',
            # 'L2DiaND_XCO2_2017_downsampled_eof_removed_aligned.p',
            # 'L2DiaND_XCO2_2018_downsampled_eof_removed_aligned.p',
            # 'L2DiaND_XCO2_2019_downsampled_eof_removed_aligned.p',
            'L2DiaND_XCO2_2020_downsampled_eof_removed_aligned.p',
            'L2DiaND_XCO2_2021_downsampled_eof_removed_aligned.p',
            ]
        test_list = [
                'L2DiaND_TCCON_Lnd_matched_2015.p',
                'L2DiaND_TCCON_Lnd_matched_2016.p',
                'L2DiaND_TCCON_Lnd_matched_2017.p',
                'L2DiaND_TCCON_Lnd_matched_2018.p',
                'L2DiaND_TCCON_Lnd_matched_2019.p',
                'L2DiaND_TCCON_Lnd_matched_2020.p',
            ]
    
        train_files = []
        test_files = []
        for file in train_list:
            train_files.append(os.path.join(data_dir, file))
        for file in test_list:
            test_files.append(os.path.join(data_dir, file))

    # TODO open all tccon pkls, get xco2acos and xco2tccon and caluculate the rmse.
    xco2acos = []
    xco2tccon = []
    for file in test_files:
        with open(file, 'rb') as file:
            data = pickle.load(file)
        _xco2acos = (data['states'][:, 20].reshape(-1, 1))
        _xco2acos = _xco2acos * 1e6
        _xco2tccon = data['xco2tccon']
        xco2acos.append(_xco2acos)
        xco2tccon.append(_xco2tccon)
    xco2acos = np.concatenate(xco2acos)
    xco2tccon = np.concatenate(xco2tccon)    
    rmse = np.sqrt(np.mean((xco2acos - xco2tccon) ** 2))
    print(f'ACOS v. TCCON RMSE : {rmse}')




    dm = RetrievalDataModule(train_files, test_files, batch_size=args.batch_size,
                             normalize=True, 
                             n_test=None, 
                             val_split=0.2, 
                             use_convolutions=False, 
                             normalize_file='retrieval_normalize_stats.pkl', 
                             pca_bands=args.pca_bands, pca_components=args.pca_components,)
    dm.setup('fit')

    INPUT_DIM = dm.get_input_shape()
    OUTPUT_DIM = dm.get_target_shape()
    INPUT_DIM = INPUT_DIM[1]
    OUTPUT_DIM = OUTPUT_DIM[1]

    # get number of gpus available
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    if len(available_gpus) >= args.num_devices:
        DEVICE = 'cuda'
        NUM_DEVICES = args.num_devices
    else:
        DEVICE = 'cpu'
        NUM_DEVICES = 0



    if model_type == 'CondDiff':
        M = 1
        conditional_mean_epochs = args.conditional_mean_epochs
        guidance_model_epochs = args.guidance_model_epochs
        lr_backbone = args.lr_backbone
        lr_guidance = args.lr_guidance
        n_z_samples = args.n_z_samples
        n_steps = args.n_steps
        beta_start = args.beta_start
        beta_end = args.beta_end
        cat_x = args.cat_x  # condition on input x through concatenation
        cat_y_pred = args.cat_y_pred  # condition on y_0_hat

        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM

        n_hidden_backbone = args.n_hidden_backbone
        n_hidden_guidance = args.n_hidden_guidance

        beta_schedule = args.beta_scheduler

        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=args.min_delta_cond, patience=args.patience_cond, mode='min', verbose=True)
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor="val_loss")

        preds_mean = []
        preds_epi = []
        preds_uct = []

        save_dir = save_dir + 'Diffusion/'
        for m in range(M):
            print('Training ensemble member : ' + str(m))
            network = NonlinearMLP(n_inputs=x_dim, n_hidden=n_hidden_backbone, n_outputs=y_dim, activation=nn.LeakyReLU())

            cond_mean_model = DeterministicRegression(
                model=network, optimizer=partial(torch.optim.Adam, lr=lr_backbone), loss_fn=nn.MSELoss(), 
            )
            trainer = Trainer(
                max_epochs=conditional_mean_epochs,  # number of epochs we want to train
                accelerator="gpu",
                enable_checkpointing=True,
                enable_progress_bar=True,
                callbacks=[early_stopping_callback,checkpoint_callback],
            )
            trainer.fit(cond_mean_model, dm)
            # save the model
            torch.save(network.state_dict(), os.path.join(save_dir, f'pre_conditioner_{m}.pt'))

            early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=args.min_delta_guidance, patience=args.patience_guidance, mode='min', verbose=True)
            # lr_monitor = LearningRateMonitor(logging_interval='step')
            checkpoint_callback = ModelCheckpoint(monitor="val_loss")

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
                guidance_optim=partial(torch.optim.Adam, lr=lr_guidance),
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                n_steps=n_steps,
                n_z_samples=n_z_samples,
            )
            diff_trainer = Trainer(
                max_epochs=guidance_model_epochs,  # number of epochs we want to train
                devices=[NUM_DEVICES],
                accelerator=DEVICE,
                enable_checkpointing=True,
                enable_progress_bar=True,
                callbacks=[early_stopping_callback,checkpoint_callback],
            )
            diff_trainer.fit(card_model, dm)
            # save the model
            torch.save(guidance_model.state_dict(), os.path.join(save_dir, f'card_model_{m}.pt'))

            print('Training complete')





            if M == 1:
                dm.setup('test')
                test_loader = dm.test_dataloader()
                #  load the model from checkpoint
               
                preds = card_model.predict_step(test_loader.dataset.X)
                y_test = dm.test_dataset.y

                with open('retrieval_normalize_stats.pkl', 'rb') as file:
                    norm_params = pickle.load(file)

                y_mean = norm_params['y_mean']
                y_std = norm_params['y_std']

                pred_unnormed = preds['pred'] * y_std + y_mean
                pred_uct = preds['pred_uct'] * y_std
                pred_uct = pred_uct.detach().numpy()
                pred_unnormed = pred_unnormed.detach().numpy()
                y_test = y_test.numpy()
                pred_unnormed = pred_unnormed * 1e6

                print(pred_unnormed[0])
                print(y_test[0])

                rmse = np.sqrt(np.mean((pred_unnormed - y_test) ** 2))
                print(f'RMSE : {rmse}')

                results = {
                    'pred': pred_unnormed,
                    'uct': pred_uct,
                    'y_test': y_test,
                }
                with open('diffustion_retrieval_results.pkl', 'wb') as file:
                    pickle.dump(results, file)

                print('done..')


    # TODO :  load modesl and test. Save results


    # TODO : Add Evidential, SNGP, DeepEnsemble, MLE, MCSample
    if model_type == 'Evidential':
        save_dir = save_dir + 'Evidential/'
        epochs = args.evidential_epochs

        lr = args.lr
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM

        
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=args.min_delta, patience=args.patience, mode='min', verbose=True)
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor="val_loss")

        network = MLP(n_inputs=x_dim, n_hidden=args.n_hidden_evidential, n_outputs=4, activation_fn=nn.LeakyReLU())
        der_model = DER(network, optimizer=partial(torch.optim.Adam, lr=lr))
        trainer = Trainer(
            max_epochs=epochs,  # number of epochs we want to train
            enable_checkpointing=True,
            enable_progress_bar=True,
            callbacks=[early_stopping_callback, checkpoint_callback],
            accelerator=DEVICE,
            devices=[NUM_DEVICES],
        )
        trainer.fit(der_model, dm)

        # save the model
        torch.save(network.state_dict(), os.path.join(save_dir, 'der_model.pt'))

        print('Training complete')

        dm.setup('test')
        test_loader = dm.test_dataloader()
        preds = der_model.predict_step(test_loader.dataset.X)
        y_test = dm.test_dataset.y

        with open('retrieval_normalize_stats.pkl', 'rb') as file:
            norm_params = pickle.load(file)

        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']

        pred_unnormed = preds['pred'] * y_std + y_mean
        pred_uct = preds['pred_uct'] * y_std
        pred_uct = pred_uct.detach().numpy()
        pred_unnormed = pred_unnormed.detach().numpy()
        y_test = y_test.numpy()
        pred_unnormed = pred_unnormed * 1e6

        print(pred_unnormed[0])
        print(y_test[0])

        rmse = np.sqrt(np.mean((pred_unnormed - y_test) ** 2))
        print(f'RMSE : {rmse}')

        results = {
            'pred': pred_unnormed,
            'uct': pred_uct,
            'y_test': y_test,
        }

        with open('evidential_retrieval_results.pkl', 'wb') as file:
            pickle.dump(results, file)

        print('done..')




    if model_type == 'SNGP':
        save_dir = save_dir + 'SNGP/'
        epochs = args.sngp_epochs
        lr = args.lr
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_features = args.n_features
        depth = args.depth

        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = args.min_delta, patience=args.patience, mode='min', verbose=True)
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=True)

        feature_extractor = FCResNet(input_dim=x_dim, features=n_features, depth=depth)
        sngp = SNGPRegression(
            feature_extractor=feature_extractor,
            loss_fn=torch.nn.MSELoss(),
            optimizer=partial(torch.optim.Adam, lr=3e-3),
        )
        trainer = Trainer(
            max_epochs=epochs,  # number of epochs we want to train
            enable_checkpointing=True,
            enable_progress_bar=True,
            callbacks=[early_stopping_callback, checkpoint_callback],
            accelerator="cpu"
        )
        trainer.fit(sngp, dm)

        # save the model
        torch.save(sngp.feature_extractor.state_dict(), os.path.join(save_dir, 'sngp_feature_model.pt'))

        print('Training complete')

        dm.setup('test')
        test_loader = dm.test_dataloader()
        preds = sngp.predict_step(test_loader.dataset.X)
        y_test = dm.test_dataset.y

        with open('retrieval_normalize_stats.pkl', 'rb') as file:
            norm_params = pickle.load(file)

        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']

        pred_unnormed = preds['pred'] * y_std + y_mean
        pred_uct = preds['pred_uct'] * y_std
        pred_uct = pred_uct.detach().numpy()
        pred_unnormed = pred_unnormed.detach().numpy()
        y_test = y_test.numpy()
        pred_unnormed = pred_unnormed * 1e6

        print(pred_unnormed[0])
        print(y_test[0])

        rmse = np.sqrt(np.mean((pred_unnormed - y_test) ** 2))
        print(f'RMSE : {rmse}')

        results = {
            'pred': pred_unnormed,
            'uct': pred_uct,
            'y_test': y_test,
        }

        with open('sngp_retrieval_results.pkl', 'wb') as file:
            pickle.dump(results, file)

        print('done..')
        


    if model_type == 'DeepEnsemble':
        save_dir = save_dir + 'DeepEnsemble/'
        M = args.M
        epochs = args.deep_ensemble_epochs
        lr = args.lr
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_hidden = args.n_hidden_deep_ensemble
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = args.min_delta, patience=args.patience, mode='min', verbose=True)
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=True)

        
        trained_models_nll = []
        for m in range(M):
            mlp_model = MLP(n_inputs = x_dim, n_hidden=n_hidden, n_outputs=2, activation_fn=nn.LeakyReLU())
            ensemble_member = MVERegression(
                mlp_model, optimizer=partial(torch.optim.Adam, lr=lr), burnin_epochs=0
            )
            trainer = Trainer(
                max_epochs=epochs,
                enable_checkpointing=True,
                enable_progress_bar=True,
                callbacks=[early_stopping_callback, checkpoint_callback],
                accelerator=DEVICE,
            )
            trainer.fit(ensemble_member, dm)
            save_path = os.path.join(save_dir, f"model_nll_{m}.ckpt")
            trainer.save_checkpoint(save_path)
            trained_models_nll.append({"base_model": ensemble_member, "ckpt_path": save_path})
        deep_ens_nll = DeepEnsembleRegression(M,trained_models_nll)

        print('Training complete')

        dm.setup('test')
        test_loader = dm.test_dataloader()
        preds = deep_ens_nll.predict_step(test_loader.dataset.X)
        y_test = dm.test_dataset.y

        with open('retrieval_normalize_stats.pkl', 'rb') as file:
            norm_params = pickle.load(file)

        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']

        pred_unnormed = preds['pred'] * y_std + y_mean
        pred_uct = preds['pred_uct'] * y_std
        pred_uct = pred_uct.detach().numpy()
        pred_unnormed = pred_unnormed.detach().numpy()
        y_test = y_test.numpy()
        pred_unnormed = pred_unnormed * 1e6

        print(pred_unnormed[0])
        print(y_test[0])

        rmse = np.sqrt(np.mean((pred_unnormed - y_test) ** 2))
        print(f'RMSE : {rmse}')

        results = {
            'pred': pred_unnormed,
            'uct': pred_uct,
            'y_test': y_test,
        }

        with open('deep_ensemble_retrieval_results.pkl', 'wb') as file:
            pickle.dump(results, file)

        print('done..')


        
    if model_type == 'MLE':
        # TODO : add MLE ensemble
        save_dir = save_dir + 'MLE/'
        epochs = args.mle_epochs
        lr = args.lr
        x_dim = INPUT_DIM
        y_dim = OUTPUT_DIM
        n_hidden = args.n_hidden_mle

        save_dir = save_dir + 'MLE/'
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=args.min_delta, patience=args.patience, mode='min', verbose=True)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, verbose=True)

        mlp_model = MLP(n_inputs=x_dim, n_hidden=n_hidden, n_outputs=y_dim, activation_fn=nn.LeakyReLU())
        MLE = DeterministicRegression( mlp_model, optimizer=partial(torch.optim.Adam, lr=lr), loss_fn=nn.MSELoss())
        trainer = Trainer(
            max_epochs=epochs,
            enable_checkpointing=True,
            enable_progress_bar=True,
            callbacks=[early_stopping_callback, checkpoint_callback],
            accelerator=DEVICE,
            devices=[NUM_DEVICES],
        )
        trainer.fit(MLE, dm)

        # save the model
        torch.save(mlp_model.state_dict(), os.path.join(save_dir, 'mle_model.pt'))

        print('Training complete')

        dm.setup('test')
        test_loader = dm.test_dataloader()
        preds = MLE.predict_step(test_loader.dataset.X)
        y_test = dm.test_dataset.y

        with open('retrieval_normalize_stats.pkl', 'rb') as file:
            norm_params = pickle.load(file)

        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']

        pred_unnormed = preds['pred'] * y_std + y_mean

        pred_unnormed = preds['pred'] * y_std + y_mean
        pred_unnormed = pred_unnormed.detach().numpy()
        y_test = y_test.numpy()
        pred_unnormed = pred_unnormed * 1e6

        print(pred_unnormed[0])
        print(y_test[0])

        rmse = np.sqrt(np.mean((pred_unnormed - y_test) ** 2))
        print(f'RMSE : {rmse}')

        results = {
            'pred': pred_unnormed,
            'y_test': y_test,
        }

        with open('mle_retrieval_results.pkl', 'wb') as file:
            pickle.dump(results, file)

        print('done..')

    if model_type == 'MCSample':
        save_dir = save_dir + 'MCSample/'
        # TODO : Load Diffusion model
        # TODO : draw samples through MC sampling
        # TODO : KDE on samples to get predictive distribution
        # TODO : Save jpgs of predictive distribution, Gaussian and non-Gaussian examples






if __name__ == '__main__':
    args = parser()
    main(args)











