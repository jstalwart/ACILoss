#from persim import wasserstein, bottleneck, PersistenceEntropy
from transformers import InformerConfig, InformerForPrediction, AutoformerConfig, AutoformerForPrediction
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from .Transformer.Transformer import EncoderPart, DecoderPart
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from .Informer.Encoder import EncoderLayer
from .Informer.Decoder import DecoderLayer
from .Informer.Encoder import InfEncoder
from .Informer.Decoder import InfDecoder
from torch.utils.data import DataLoader
from .Informer.Attention import *
from .Transformer.Head import *
from .Dataset import *
import torch.nn as nn
from tqdm import tqdm
from .utils import *
from .ACI import *
import numpy as np
import random
import torch
import time
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class Experiment:
    '''
    Parameters:
        - name (str): the name of the experiment
        - model (str): the name of the TSF model employed. 
        - input_size (int): the timestamps considered for prediction
        - emb_size (int): the embedding size. 
        - output_size (int): the prediction horizon. 
        - loss1 (str): the first Loss function employed. It can be None. 
        - loss2 (str): the second Loss function emplyed. Current version only allows for MSE and NLL. 
        - seed (int): the random seed used in the experiment. 
        - device (str): the deviced for the execution. 
        - enc_depth (int): the number of layers in the encoder. 
        - dec_depth (int): the number of layers in the decoder. 
        - heads (int): the number of heads in the attention modules. 
        - batch_size (int): the number of observations per batch. 
        - dropout (float): the dropout present in the model. 
        - dataset (str): the name of dataset employed. 
        - freq (str): the frequence for timestamps in the series. 
        - exogenous (dict): the exogenous variables to account for each dataset. 
        - train_data (DataLoader): the data employed for training. 
        - test_data (DataLoader): the data employed for testing. 
        - val_data (DataLoader): the data employed for validation. 
        - encoder (nn.Model): the model employed for encoding the series. 
        - decoder (nn.Model): the model employed for decoding the series. 
    '''

    def __init__(self,
                 name:str,
                 dataset: str,
                 input_size:int,
                 emb_size:int,
                 output_size:int,
                 model:str,
                 loss1:str,
                 loss2:str,
                 seed:int,
                 device:str = None,
                 batch_size:int = 32,
                 enc_depth:int = 8,
                 dec_depth:int = 8,
                 heads: int = 5,
                 dropout:float=0.0):
        '''
        Initialises the experiment. 

        Args:
            - name (str): the experiment name. 
            - dataset (str): the dataset to forecast. 
            - input_size (int): the input data size.
            - emb_size (int): the embedding size. 
            - output_size (int): the output data size. 
            - model (str): the TSF model employed. Current version only supports for Transformer and Informer. 
            - loss1 (str): the frist loss criterion. Current version supports ACI, SDM or regular. 
            - loss2 (str): the second loss criterion. Current version supports MSE or NLL. 
            - seed (int): for experiment replication. 
            - device (str): the device to allocate the model. If None, it uses gpu if it can.
            - batch_size (int): the batch size. 
            - enc_depth (int): the encoder depth. Default is 8.
            - dec_depth (int): the decoder depth. Default is 8. 
            - heads (int): the number of heads in the attention modules. Default is 5. 
            - dropout (float): the dropout for all the model. Default is 0.0. 
        '''
        self.name = name
        self.model = model.lower()
        self.input_size = input_size
        self.emb_size = emb_size
        self.output_size=output_size
        self.loss1 = loss1.upper()
        self.loss2 = loss2.upper()
        self.seed = seed
        self.device = device if device != None else "cuda" if torch.cuda.is_available() else "cpu"
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.heads = heads
        self.batch_size = batch_size
        self.dropout = dropout

        losses = ["ACI", "", "SDM"]
        assert self.loss1 in losses, f"Loss {loss1} is ill-defined. Current version only supports 'ACI', 'SDM' or None."
        losses = ["MSE", "NLL"]
        assert self.loss2 in losses, f"Loss {loss2} is ill-defined. Current version only supports 'MSE' or 'NLL'."
        models = ["transformer", "informer", "autoformer"]
        assert self.model in models, f"Model {model} is ill-defined. Current version only supports {', '.join(models)}."
        select_seed(self.seed)
        self.prepare_data(dataset)
        self.prepare_encoder()
        self.prepare_decoder()

    def prepare_data(self, dataset):
        '''
        Prepares the dataset for using. 
        ---
        Args:
            - dataset (str): the name of the dataset to use.        
        '''
        self.dataset = dataset.upper()

        datasets = ["M1", "M2", "H1", "H2", "ECL", "PC", "TFF", "WTH", "ER", "ILI"]
        assert self.dataset in datasets, f"Dataset {dataset} is not implemented. Current version only allows for {', '.join(datasets)}."
        self.freq = dict(zip(datasets, ["t", "t", "h", "h", "t", "h", "h", "h", "d", "w"]))[self.dataset]
        real_datasets = dict(zip(datasets, ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Pedestrian", "Traffic", "Weather", "ER", "ILI"]))
        endogenous = dict(zip(datasets, ["OT"]*4+["MT_320", "value", "T407", "temperature", "Singapore", "ILITOTAL"]))
        exogenous = [["HUFL","HULL","MUFL","MULL","LUFL","LULL","fourier_sin_order1", "fourier_cos_order1"]]*4 + [
                     [f"MT_{i+1:03}" for i in range(370) if i+1 != 320], 
                     [f"T_{i+1}" for i in range(66) if i+1 != 9],
                     [f"T_{i+1}" for i in range(861) if i+1 != 407],
                     ["total_cloud_cover", "dewpoint_temperature", "surface_solar_radiation", "wind_speed", "mean_sea_level_pressure", "relative_humidity", "surface_thermal_radiation"],
                     ["Australia", "UK", "Canada", "Switzerland", "China", "Japan", "New Zeland"] ,
                     ["NUM. OF PROVIDERS", "TOTAL PATIENTS"]
        ]
        self.exogenous = dict(zip(datasets, exogenous))     

        assert self.dataset in datasets, f"Dataset {dataset} is ill-defined. Current version only supports {','.join(datasets)}."

        train_data = TSDataset(path = f"../00-Data/{real_datasets[self.dataset]}.csv", 
                                 endogenous = endogenous[self.dataset], 
                                 exogenous = self.exogenous[self.dataset],
                                 window_size=self.input_size, 
                                 horizon=self.output_size, 
                                 mode = "train")
        test_data = train_data.copy(mode="test")
        val_data = train_data.copy(mode="val")

        self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_data = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

    def prepare_encoder(self, **kwargs):
        '''
        Defines the encoder. 
        '''                             
        if self.model == "transformer":
            self.encoder = EncoderPart(input_channels = len(self.exogenous[self.dataset])+1, 
                                       d_model = self.emb_size, 
                                       depth = self.enc_depth, 
                                       n_heads = self.heads, 
                                       forward_expansion = 4, 
                                       embed_type = "fixed", 
                                       freq = "h", 
                                       dropout = self.dropout).to(self.device)
        elif self.model == "informer":
            self.encoder = InfEncoder(input_features = len(self.exogenous[self.dataset])+1, 
                                      d_model = self.emb_size, 
                                      attn_layers = [EncoderLayer(
                                          AttentionLayer(
                                              ProbAttention(False, 5, attention_dropout=0.0, output_attention=False),
                                              self.emb_size, self.heads, mix=False),
                                          self.emb_size,
                                          self.emb_size,
                                          dropout=0.0,
                                          activation="gelu") for l in range(self.enc_depth)], 
                                      conv_layers=None, 
                                      norm_layer = torch.nn.LayerNorm(self.emb_size),
                                      embed_type="fixed", 
                                      freq = self.freq, 
                                      dropout = self.dropout).to(self.device)
        
    def load_encoder(self,
                     state:dict = None,
                     path:str = None,
                     **kwargs):
        '''
        Initializes the encoder and loads its parameters. 
        ---
        Args:
            - path (str): the path of the file where the parameter values are storaged. 
        '''
        path = f"./Models/{self.name}/encoder_{self.seed}.pt" if path is None else path
        self.prepare_encoder(**kwargs)
        print("\n---- Loading encoder ----")
        state = torch.load(path, map_location=self.device) if state==None else state
        self.encoder.load_state_dict(state)
    
    def prepare_decoder(self, **kwargs):
        '''
        Defines the decoder. 
        '''
        if self.model == "transformer":
            self.decoder = DecoderPart(input_channels = len(self.exogenous[self.dataset]), 
                                       output_size = self.output_size, 
                                       d_model = self.emb_size, 
                                       depth = self.dec_depth, 
                                       n_heads_self=self.heads, 
                                       n_heads_cross = self.heads, 
                                       forward_expansion = 4, 
                                       embed_type="fixed", 
                                       freq = self.freq, 
                                       dropout = self.dropout).to(self.device)
        elif self.model == "informer":
            self.decoder = InfDecoder(input_channels=len(self.exogenous[self.dataset]), 
                                      output_features = self.output_size, 
                                      d_model = self.emb_size, 
                                      layers = [
                                        DecoderLayer(
                                            AttentionLayer(
                                                ProbAttention(True, 5, attention_dropout=self.dropout, output_attention=False),
                                                self.emb_size,
                                                self.heads, 
                                                mix = True), 
                                            AttentionLayer(
                                                FullAttention(False, 5, attention_dropout=self.dropout, output_attention=False),
                                                self.emb_size, 
                                                self.heads, 
                                                mix = False),
                                        d_model = self.emb_size, 
                                        d_ff = self.emb_size, 
                                        dropout = self.dropout,
                                        activation = "gelu") for l in range(self.dec_depth)], 
                                      norm_layer=nn.LayerNorm(self.emb_size), 
                                      embed_type="fixed", 
                                      freq = self.freq, 
                                      dropout = self.dropout).to(self.device)
            
    def load_decoder(self,
                   state:dict = None):
        '''
        Initializes the model  and loads its parameters. 
        ---
        Args:
            - state: dictionary for the model's parameters. If None, the model tries loading the model from a file.
        '''
        print("\n---- Loading decoder ----")
        state = torch.load(f"./Models/{self.name}/decoder_{self.seed}.pt", map_location=self.device) if state==None else state
        self.decoder.load_state_dict(state)
        
    def save_log(self, 
                 optimizer, 
                 epoch:int, 
                 train_loss:float, 
                 test_loss:float, 
                 time_epoch:float):
        """
        Saves the log for the model. 
        ---
        Args:
            - optimizer: the optimizer used. Current version only supports Adam.
            - epoch (int): appointing to the current epoch.
            - train_RMSE (float): RMSE obtained during training.. 
            - test_RMSE (float): RMSE obtained during testing.
            - time_epoch (float): training time in seconds. 
        """

        f = open(f"./Logs/{self.name}/{self.seed}.log", "a")
        f.write("[Epoch {}] Train Loss: {:e} Test Loss: {:e} Time {:.10e} Learning rate: {:1e}\n".format(
            epoch + 1, train_loss, test_loss, time_epoch, get_lr(optimizer),
            ))
        f.close()

    def train(self, optimizer, scaler=.25, **kwargs):
        '''
        Trains the model for one epoch. 
        ---
        Args:
            - optimizer (algorithm): the algoritm for optimizing the loss function. 
            - scaler (float): the scaler for computing the loss function. 
        ---
        Returns:
            - total_loss (float): the computed loss for all batches. 
            - total_time (float): the time in seconds it needed to train the epoch.
        '''
        start = time.time()
        total_loss = 0.0
        num_batches = 0

        if self.loss1 == "ACI":
            first_criterion = ACILoss()
        elif self.loss1 == "SDM":
            first_criterion = SDMLoss()
            
        if self.loss2 == "NLL":
            second_criterion = nn.GaussianNLLLoss()
        elif self.loss2 == "MSE":
            second_criterion = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()
        for batch in tqdm(self.train_data, desc="Training model"):
            optimizer.zero_grad()
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            z, t1, t2 = batch["z"].to(self.device), batch["t1"].to(self.device), batch["t2"].to(self.device)
            dec_self_mask = batch["mask"].to(self.device)

            
            embedding = self.encoder(x=x,
                                     x_mark = t1)
            
            crit1 = first_criterion(embedding, x) if self.loss1!="" else 0.0

            output, variance = self.decoder(y = z, 
                                            y_mark = t2,
                                            x = embedding, 
                                            self_mask = dec_self_mask,
                                            cross_mask = None)
            if self.loss2 == "MSE":
                crit2 = second_criterion(output, y)
            elif self.loss2 == "NLL":
                crit2 = second_criterion(output, y, variance)

            loss = scaler * crit1 + (1-scaler) * crit2 if self.loss1 != "" else crit2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches, time.time() - start
    
    def test(self, 
             partition : str,
             scaler: float = .25,
             metrics : list = ["RMSE", "MAE", "MAPE"]):
        '''
        Tests the model. 
        ---
        Args:
            - partition (str); the data partition where to test the model. Current version only supports 'train', 'test' or 'val'. 
            - metrics (list of str): the metrics to evaluate the model. Default are RMSE and MAPE. 
        ---
        Returns:
            - results: dicts with metric as keys and results as values.  
        '''
        assert partition.lower() in ["train", "test", "val"], f"{partition} partition is not allowed. Current version only supports 'train', 'test' or 'val' partition."
        available_metrics = ["LOSS", "RMSE", "MAPE", "MAE"]
        for metric in metrics:
            assert metric.upper() in available_metrics, f"{metric} metric is not implemented. Current version only supports {', '.join(available_metrics)}."
        partition = partition.lower()
        metrics = [metric.upper() for metric in metrics]
        part = self.train_data if partition=="train" else self.test_data if partition == "test" else self.val_data

        self.encoder.eval()
        self.decoder.eval()

        if self.loss1 == "ACI":
            first_criterion = ACILoss()
        elif self.loss1 == "SDM":
            first_criterion = SDMLoss()
        
        if self.loss2 == "NLL":
            second_criterion = nn.GaussianNLLLoss()
        elif self.loss2 == "MSE":
            second_criterion = nn.MSELoss()

        total_loss= 0.0
        num_batches = 0
        y_pred = []
        y_true = []
        emb_list = []
        x_list = []

        with torch.no_grad():      
            for batch in tqdm(part, desc="Testing model"):
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                z, t1, t2 = batch["z"].to(self.device), batch["t1"].to(self.device), batch["t2"].to(self.device)
                dec_self_mask = batch["mask"].to(self.device)
                std, median = batch["std"].to(self.device), batch["median"].to(self.device)

                embedding = self.encoder(x=x,
                                        x_mark = t1)
                
                crit1 = first_criterion(embedding, x) if self.loss1!="" else 0.0
                
                #embedding = embedding.squeeze(0).unsqueeze(1)

                output, variance = self.decoder(y = z, 
                                                y_mark = t2,
                                                x = embedding, 
                                                self_mask = dec_self_mask,
                                                cross_mask = None)
                if self.loss2 == "MSE":
                    crit2 = second_criterion(output, y)
                elif self.loss2 == "NLL":
                    crit2 = second_criterion(output, y, variance)

                loss = scaler * crit1 + (1-scaler) * crit2 if self.loss1 != "" else crit2
                
                total_loss += loss.item()
                num_batches += 1

                y_denorm = (y*(std+1e-8))+median
                o_denorm = (output*(std+1e-8))+median

                x_list.extend(x.detach().cpu().tolist())
                y_true.extend(y_denorm.detach().cpu().tolist())
                emb_list.extend(embedding.squeeze(dim=0).detach().cpu().tolist())
                y_pred.extend(o_denorm.detach().cpu().tolist())

        results = {}
        for metric in metrics:
            if metric == "LOSS":
                results["Loss"] = total_loss/num_batches
            if metric == "RMSE":
                results["RMSE"] = RMSE(y_true, y_pred)
            if metric == "MAE":
                results["MAE"] = MAE(y_true, y_pred)
            if metric == "MAPE":
                results["MAPE"] = MAPE(y_true, y_pred)
            if metric == "ACI":
                aci = ACILoss()
                results["ACI"] = aci(torch.Tensor(emb_list), torch.Tensor(x_list)).item()
            if metric == "SDM":
                sdm = SDMLoss()
                results["SDM"] = sdm(torch.Tensor(emb_list), torch.Tensor(x_list)).item()
        
        return results
    
    def n_params(self, model):
        '''
        Returns the number of parameters for the given model. 
        ---
        Args:
            - model (nn.Module): the model to obain the n_params. 
        ---
        Returns:
            - n_params (int): the number of parameters in the model. 
        '''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def evaluation(self, 
                   params_encoder: dict = None, 
                   params_decoder: dict = None, 
                   best_epoch: int = None, 
                   best_train_loss: float = None, 
                   best_test_loss: float = None, 
                   training_times: list = [],
                   metrics = ["RMSE", "MAPE", "MAE"],
                   **kwargs):
        '''
        Evaluates the model using the validation partition.
        ---
        Args:
        - params_encoder (dict): state_dict of the encoder parameters. 
        - params_decoder (dict): state_dict of the decpder parameters. 
        - best_epoch (int): the epoch achieving the best test. 
        - best_train_loss (float): the loss obtained in the best_epoch.
        - best_test_loss (float): the best loss obtained in test.
        - training_times (list of float): a list of the training times. 
        - metrics (list of str): the metrics to print in the evaluation scheme. 
        ---
        Returns:
        - None
        '''
        # Load model
        self.load_encoder(state=params_encoder)
        self.load_decoder(state=params_decoder)

        print("Parameters: {:.2e}".format(self.n_params(self.encoder)+self.n_params(self.decoder)))

        if best_train_loss != None:
            print("Train Loss: {:.2e} in epoch {}.".format(best_train_loss, best_epoch))
        elif "train_loss" in metrics:
            print("Train Loss: {:.2e}".format(self.test("train", metrics=["Loss"], **kwargs)["Loss"]))

        if best_test_loss != None:
            print("Test Loss: {:.2e} in epoch {}.".format(best_test_loss, best_epoch))
        elif "test_loss" in metrics:
            print("Test Loss: {:.2e}".format(self.test("test", metrics=["Loss"], **kwargs)["Loss"]))

        if training_times != []:
            print("Average training time by epoch: {:.2e} seconds,".format( np.mean(training_times)))
            print("Standard deviation for training time by epoch: {:.2e} seconds.".format(np.std(training_times)))

        val_dict = self.test("val", metrics=metrics, **kwargs)

        for key in val_dict.keys():
            if key.upper() in ["SMAPE", "RMSSPE", "ACCURACY", "MACRO_F1", "MICRO_F1"]:
                print(f"Validation {key}: {val_dict[key]:.2f}%")
            else:
                print(f"Validation {key}: {val_dict[key]:.4e}")

    def fit(self, 
            scaler = .25,
            epochs: int = 2000, 
            lr : float = 1e-3,
            patience : int = 30,
            scheduler_patience : int = 10,
            weight_decay = 1e-4):
        '''
        Method for training and evaluating the model. 
        ---
        Args:
            - scaler (float ranging 0 and 1): the scaler for the multiobjective loss. 
            - epochs (int): the number of iterations in training. 
            - lr (float): the learning rate for training the model. 
            - patience (int): the iterations without improvement before early stopping. Default is 30. 
            - scheduler_patience (int): the patience before dropping the learning rate. Default is 10. 
            - weight_decay (float): for the Adam Algorithm. 
        ---
        Returns:
            - None
        '''
        assert scaler <= 1 and scaler >= 0, f"Scaler {scaler} must be contained withiin 0 and 1, both inclusive."
        best_test_loss = np.inf
        count=0
        if self.model == "informer":
            optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                lr = lr, 
                weight_decay = weight_decay
            )
        else:
            optimizer = torch.optim.Adam([{"params": self.encoder.parameters(), "lr":lr},
                                        {"params": self.decoder.parameters(), "lr": lr}],
                                        weight_decay = weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            factor=.5, 
                                                            patience=scheduler_patience, 
                                                            min_lr=1e-9)
        training_times = []

        # Init Log
        f = open(f"./Logs/{self.name}/{self.seed}.log", "w")
        f.close()

        print("\n---- Start Training ----")
        # Train network
        for epoch in range(epochs):
            train_loss, train_time = self.train(optimizer=optimizer, 
                                                scaler=scaler)
            training_times.append(train_time)
            test_loss = self.test(partition="test",
                                  scaler=scaler,
                                  metrics=["Loss"])["Loss"]
            scheduler.step(test_loss)
            self.save_log(optimizer, epoch=epoch, train_loss=train_loss, test_loss=test_loss, time_epoch=train_time)

            if test_loss < best_test_loss:
                best_epoch = epoch
                best_train_loss = train_loss
                best_test_loss = test_loss
                enc_params = self.encoder.state_dict()
                dec_params = self.decoder.state_dict()
                torch.save(enc_params, f"Models/{self.name}/encoder_{self.seed}.pt")
                torch.save(dec_params, f"Models/{self.name}/decoder_{self.seed}.pt")
                count = 0
            else:
                count += 1

            if count >= patience:
                break
        
        self.evaluation(params_encoder = enc_params,
                        params_decoder = dec_params,
                        best_epoch = best_epoch,
                        best_train_loss = best_train_loss,
                        best_test_loss = best_test_loss,
                        training_times = training_times, 
                        metrics = ["RMSE", "MAE", "MAPE"])