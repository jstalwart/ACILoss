#from persim import wasserstein, bottleneck, PersistenceEntropy
from transformers import InformerConfig, InformerForPrediction, AutoformerConfig, AutoformerForPrediction
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from torch.utils.data import DataLoader
from .Wrapper import *
from .Encoder import Encoder
from .Dataset import *
import torch.nn as nn
from tqdm import tqdm
from .utils import *
from .Head import *
from .ACI import ACILoss
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
        - loss (str): the loss employed. 
        - seed (int): the random seed used in the experiment. 
        - device (str): the deviced for the execution. 
        - enc_depth (int): the number of layers in the encoder. 
        - dec_depth (int): the number of layers in the decoder. 
        - batch_size (int): the number of observations per batch. 
        - dataset (str): the name of dataset employed. 
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
                 loss:str,
                 seed:int,
                 device:str = None,
                 batch_size:int = 32,
                 enc_depth:int = 8,
                 dec_depth:int = 8):
        '''
        Initialises the experiment. 

        Args:
            - name (str): the experiment name. 
            - dataset (str): the dataset to forecast. 
            - input_size (int): the input data size.
            - emb_size (int): the embedding size. 
            - output_size (int): the output data size. 
            - mode (str): the mode ACI or regular. Default is None (regular). 
            - device (str): the device to allocate the model. If None, it uses gpu if it can.
            - seed (int): for experiment replication. 
            - batch_size (int): the batch size. 
            - enc_depth (int): the encoder depth. Default is 8.
            - dec_depth (int): the decoder depth. Default is 8. 
        '''
        self.name = name
        self.model = model.lower()
        self.input_size = input_size
        self.emb_size = emb_size
        self.output_size=output_size
        self.loss = loss.upper()
        self.seed = seed
        self.device = device if device != None else "cuda" if torch.cuda.is_available() else "cpu"
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.batch_size = batch_size

        losses = ["ACI", ""]
        assert self.loss in losses, f"Loss {loss} is ill-defined. Current version only supports 'ACI' or None."
        models = ["transformer", "informer", "autoformer"]
        assert self.model in models, f"Loss {loss} is ill-defined. Current version only supports {', '.join(models)}."
        select_seed(self.seed)
        self.prepare_data(dataset)
        self.prepare_encoder()
        self.prepare_decoder()

    def prepare_data(self, dataset):
        self.dataset = dataset.upper()

        datasets = ["M1", "M2", "ECL1", "ECL2", "ECL3", "H1", "H2", "PC", "TFF", "WTH", "ER", "ILI"]
        real_datasets = dict(zip(datasets, ["ETTm1", "ETTm2", "ECL", "ECL", "ECL", "ETTh1", "ETTh2", "Pedestrian", "Traffic", "Weather", "ER", "ILI"]))
        time_series = dict(zip(datasets, ["OT"]*2+["MT_156","MT_162", "MT_189"]+["OT"]*2+["value", "T407", "temperature", "Singapore", "ILITOTAL"]))

        assert self.dataset in datasets, f"Dataset {dataset} is ill-defined. Current version only supports {','.join(datasets)}."

        train_data = MainDataset(path = f"../00-Data/{real_datasets[self.dataset]}.csv", 
                                 endogenous = time_series[self.dataset], 
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
            self.encoder = nn.Sequential(TSEmbedding(in_features = self.input_size, 
                                                    emb_size = self.emb_size),
                                        Encoder(emb_size = self.emb_size, 
                                                depth = self.enc_depth,
                                                **kwargs)).to(self.device)
        elif self.model == "informer":
            config = InformerConfig(
                    prediction_length=self.output_size, 
                    context_length=self.input_size-1, 
                    input_size=1,
                    num_time_features=1, 
                    lags_sequence = [1],
                    scaling=None,
                    d_model = self.emb_size,
                    feature_size=4,
                )
            hf_model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly", config=config, ignore_mismatched_sizes=True)
            #hf_model = InformerForPrediction(config)
            self.encoder = hf_model.to(self.device)
        
    def load_encoder(self,
                     state:dict = None,
                     path:str = None,
                     **kwargs):
        '''
        Initializes the encoder and loads its parameters. 

        Args:
            - path (str): the path of the file where the parameter values are storaged. 
        '''
        path = f"./Models/{self.name}/encoder_{self.seed}.pt" if path is None else path
        self.prepare_encoder(**kwargs)
        print("\n---- Loading encoder ----")
        state = torch.load(path, map_location=self.device) if state==None else state
        self.encoder.load_state_dict(state)
    
    def prepare_decoder(self, **kwargs):
        if self.model == "transformer":
            self.decoder = nn.Sequential(Encoder(emb_size = self.emb_size, 
                                                depth = self.dec_depth,
                                                **kwargs),
                                        RegressionHead(emb_size=self.emb_size, 
                                                        out_size=self.output_size)).to(self.device)
        elif self.model == "informer":
            config = InformerConfig(
                prediction_length=self.output_size, 
                context_length=self.input_size, 
                num_time_features=1, 
                lags_sequence=[0],
                scaling=None
            )
            hf_model = InformerForPrediction(config=config)
            
            # Use our custom pipeline instead of nn.Sequential
            self.decoder = InformerPipeline(
                hf_informer_decoder=hf_model.model.decoder, 
                emb_size=512, 
                out_size=self.output_size
            ).to(self.device)

    def load_decoder(self,
                   state:dict = None):
        '''
        Initializes the model  and loads its parameters. 

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

        Parameters:
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
        start = time.time()
        total_loss = 0.0
        num_batches = 0

        if self.loss == "ACI":
            aci_criterion = ACILoss()
        mse_criterion = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()
        for batch in tqdm(self.train_data, desc="Training model"):
            optimizer.zero_grad()
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            if self.model == "informer":
                batch_size, seq_len = x.shape
                batch_size, pred_len = y.shape
                past_observed_mask = torch.ones(batch_size, seq_len, device=self.device)
                past_time_features = torch.zeros(batch_size, seq_len, 1, device=self.device)
                future_time_features = torch.zeros(batch_size, pred_len,  1, device=self.device)
                model_outputs = self.encoder(past_values=x, 
                                             past_time_features=past_time_features, 
                                             past_observed_mask=past_observed_mask,
                                             future_values = y,
                                             future_time_features = future_time_features
                                             )
                embedding = model_outputs.encoder_last_hidden_state
            elif self.model == "transformer":
                embedding = self.encoder(x)
            #print(embedding.shape)
            #print(model_outputs)
            if self.model == "informer":
                crit1 = aci_criterion(embedding.reshape(batch_size, embedding.shape[1]*embedding.shape[2]), x) if self.loss=="ACI" else 0.0
            elif self.model == "transformer":
                crit1 = aci_criterion(embedding.squeeze(dim=0), x) if self.loss=="ACI" else 0.0
            embedding = embedding.squeeze(0).unsqueeze(1)
            if self.model == "informer":
                crit2 = model_outputs.loss  # shape: (batch, pred_len, input_size)
                # Squeeze the last dimension if input_size=1
            else:
                output = self.decoder(embedding)
                crit2 = mse_criterion(output, y)
            loss = scaler * crit1 + (1-scaler) * crit2 if self.loss == "ACI" else crit2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches, time.time() - start
    
    def test(self, 
             partition : str,
             scaler: float = .25,
             metrics : list = ["RMSE", "MAPE"]):
        '''
        Tests the model. 

        Args:
            - partition (str); the data partition where to test the model. Current version only supports 'train', 'test' or 'val'. 
            - metrics (list of str): the metrics to evaluate the model. Default are RMSE and MAPE. 

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

        if self.loss == "ACI":
            aci_criterion = ACILoss()
        mse_criterion = nn.MSELoss()

        total_loss= 0.0
        num_batches = 0
        y_pred = []
        y_true = []
        emb_list = []
        x_list = []

        with torch.no_grad():
            for batch in tqdm(part, desc="Testing model"):
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                std, median = batch["std"].to(self.device), batch["median"].to(self.device)

                if self.model == "informer":
                    batch_size, seq_len = x.shape
                    _, pred_len = y.shape
                    past_observed_mask = torch.ones(batch_size, seq_len, device=self.device)
                    past_time_features = torch.zeros(batch_size, seq_len, 1, device=self.device)
                    future_time_features = torch.zeros(batch_size, pred_len,  1, device=self.device)
                    model_outputs = self.encoder(past_values=x, 
                                                past_time_features=past_time_features, 
                                                past_observed_mask=past_observed_mask,
                                                future_values = y,
                                                future_time_features = future_time_features
                                                )
                    embedding = model_outputs.encoder_last_hidden_state
                else:
                    embedding = self.encoder(x)
                
                if self.model == "informer":
                    crit1 = aci_criterion(embedding.reshape(batch_size, embedding.shape[1]*embedding.shape[2]), x) if self.loss=="ACI" else 0.0
                elif self.model == "transformer":
                    crit1 = aci_criterion(embedding.squeeze(dim=0), x) if self.loss=="ACI" else 0.0
                embedding = embedding.squeeze(0).unsqueeze(1)
                if self.model == "informer":
                    crit2 = model_outputs.loss  # shape: (batch, pred_len, input_size)
                    #print(model_outputs.last_hidden_state.shape)
                    outputs_generated = self.encoder.generate(
                        past_values=x, 
                        past_time_features=past_time_features, 
                        past_observed_mask=past_observed_mask,
                        future_time_features=future_time_features
                    )
                    output = outputs_generated.sequences.mean(dim=1)
                    # Squeeze the last dimension if input_size=1
                else:
                    output = self.decoder(embedding)
                    crit2 = mse_criterion(output, y)
                loss = scaler * crit1 + (1-scaler) * crit2 if self.loss == "ACI" else crit2
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
        
        return results
    
    def n_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def evaluation(self, 
                   params_encoder: dict = None, 
                   params_decoder: dict = None, 
                   best_epoch: int = None, 
                   best_train_loss: float = None, 
                   best_test_loss: float = None, 
                   training_times: list = [],
                   metrics = ["RMSE", "MAPE"],
                   **kwargs):
        '''
        Evaluates the model using the validation partition.

        Args:
        - params (dict): state_dict of the model parameters. 
        - best_epoch (int): the epoch achieving the best test. 
        - best_train_RMSE (float): the RMSE obtained in the best_epoch.
        - best_test_RMSE (float): the best RMSE obtained in test. std
        - training_times (list): a list of the training times. 

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

        Args:
            - epochs (int): the number of iterations in training. 
            - lr (float): the learning rate for training the model. 
            - patience (int): the iterations without improvement before early stopping. Default is 30. 
            - scheduler_patience (int): the patience before dropping the learning rate. Default is 10. 
            - weight_decay (float): for the Adam Algorithm. 

        Returns:
            - None
        '''
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
                        metrics = ["RMSE", "MAPE"])