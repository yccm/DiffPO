import numpy as np
import torch
import torch.nn as nn
from src.diff_model import diff_CSDI
import yaml


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        # keep the __init__ the same
        super().__init__()
        self.device = device
        self.target_dim = config["train"]["batch_size"]
        

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        self.cond_dim = config["diffusion"]["cond_dim"]
        self.mapping_noise = nn.Linear(2, self.cond_dim)

        if self.is_unconditional == False:
            self.emb_total_dim += 1

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = 0.5
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        side_info = cond_mask
        
        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, gt_mask, side_info, is_train, set_t=-1, propnet=None
    ):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data[:, :, 1:3])
        noisy_data = (current_alpha**0.5) * observed_data[:, :, 1:3] + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        a = observed_data[:,:,0].unsqueeze(2)
        x = observed_data[:,:,5:]
        cond_obs = torch.cat([a,x], dim=2)

        noisy_target = self.mapping_noise(noisy_data) 
        diff_input = cond_obs + noisy_target

        predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)

        target_mask = gt_mask - cond_mask
        target_mask = target_mask.squeeze(1)[:,1:3]

        noise = noise.squeeze(1)
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()

        x_batch = observed_data[:, :, 5:]
        if len(x_batch.shape) > 2: 
            x_batch = x_batch.squeeze() 
        if len(x_batch.shape) < 2:
            x_batch = x_batch.unsqueeze(1)
        t_batch = observed_data[:, :, 0].squeeze()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        propnet = propnet.to(device)


        pi_hat = propnet.forward(x_batch.float())
        pi_hat = torch.clamp(pi_hat, min=0.05, max=0.95) 

        weights = (t_batch / pi_hat[:, 1]) + ((1 - t_batch) / pi_hat[:, 0])
        weights = weights.reshape(-1, 1, 1) 

        # Clip weights for stable training
        weights = torch.clamp(weights, min=0.5, max=3) 
        weights = weights / weights.mean() #normalize

        loss = (weights * (residual ** 2)).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)
        else:
            a = observed_data[:,:,0].unsqueeze(2)
            x = observed_data[:,:,5:]
            cond_obs = torch.cat([a,x], dim=2)
            noisy_target = self.mapping_noise(noisy_data) 
            total_input = cond_obs + noisy_target
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape 

        imputed_samples = torch.zeros(B, n_samples, 2).to(self.device)

        for i in range(n_samples):
            generated_target = observed_data[:,:,1:3]

            current_sample = torch.randn_like(generated_target)

            for t in range(self.num_steps - 1, -1, -1):
                a = observed_data[:,:,0].unsqueeze(2)
                x = observed_data[:,:,5:]
                cond_obs = torch.cat([a,x], dim=2)

                noisy_target = self.mapping_noise(current_sample) 
                diff_input = cond_obs + noisy_target
 
                predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
           
                current_sample = current_sample.squeeze(1)

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

                current_sample = current_sample.unsqueeze(1)

            current_sample = current_sample.squeeze(1)
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1, propnet = None):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
    

        if is_train == 0:
            cond_mask = gt_mask.clone()
        else:
            cond_mask = gt_mask.clone()
            
            cond_mask[:, :, 1] = 0
            cond_mask[:, :, 2] = 0

        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss(observed_data, cond_mask, gt_mask, side_info, is_train, set_t=-1, propnet=propnet) if is_train == 1 else self.calc_loss_valid
        
        return loss_func

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            cond_mask[:,:,0] = 0
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class DiffPO(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(DiffPO, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"][:, np.newaxis, :]
        observed_data = observed_data.to(self.device).float()

        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        observed_tp = batch["timepoints"].to(self.device).float()

        gt_mask = batch["gt_mask"][:, np.newaxis, :]

        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )