import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import cv2

from ddrm_codes_2.models.diffusion import Model
from ddrm_codes_2.datasets import get_dataset, data_transform, inverse_data_transform
from ddrm_codes_2.functions.ckpt_util import get_ckpt_path, download
from ddrm_codes_2.functions.denoising import efficient_generalized_steps

import torchvision.utils as tvu
from torchvision.transforms import InterpolationMode, Resize

from ddrm_codes_2.guided_diffusion.unet import UNetModel
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from ddrm_codes_2.guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    
    def cosine_beta_schedule(timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype = np.float64)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, config, device=None):
        self.args = config.args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.config.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.config.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.config.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.config.args.model_dir, self.config.args.model_folder, self.config.args.model_name)
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.config.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        elif self.config.model.type == 'ddpm_ho':
            # define the 
            ckpt = os.path.join(self.args.model_dir, self.args.model_folder, self.config.model.file_name)
            mdata = torch.load(ckpt, map_location=self.device)
            
            # load the model
            if 'diffusion_model' not in mdata:
                model = Unet(dim=self.config.model.ch, dim_mults=self.config.model.ch_mult, channels=self.config.model.in_channels)
                diffusion = GaussianDiffusion(
                    model, 
                    channels= self.config.data.channels, # use 1 channel for grayscale images
                    image_size = self.config.data.image_size, # must be square images (for now)
                    timesteps = self.config.diffusion.num_diffusion_timesteps,   # number of steps
                    loss_type = 'l2'    # L1 or L2
                )
                diffusion.load_state_dict(torch.load(ckpt, map_location=self.device)['model'])
            else:
                diffusion = mdata['diffusion_model']
                diffusion.load_state_dict(mdata['model'])
                self.num_timesteps = diffusion.num_timesteps
                
            model = diffusion.denoise_fn
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.config.args, self.config
        out_folder = os.path.join(config.args.image_folder)

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        
        device_count = torch.cuda.device_count()
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(config.args.seed)

        # this gives a warning that there are too many workers. Make num_workers = 0 for now
        val_loader = data.DataLoader(
            test_dataset,
            # batch_size=config.sampling.batch_size,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            # num_workers=config.data.num_workers,
            # num_workers=os.cpu_count(),
            # num_workers=os.cpu_count()//2,
            worker_init_fn=seed_worker,
            generator=g,
        )
        

        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from ddrm_codes_2.functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif 'inp' in deg:
            from ddrm_codes_2.functions.svd_replacement import Inpainting
            sflag = False
            if deg == 'inp_lolcat':
                loaded = cv2.resize(np.load("config\\python_packages\\ddrm_codes_2\\inp_masks\\lolcat_extra.npy"),(self.config.data.image_size,self.config.data.image_size))
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = cv2.resize(np.load("inp_masks/lorem3.npy"),(self.config.data.image_size,self.config.data.image_size))
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif 'inp' in deg and '0.' in deg:
                ratio = float(deg.split('-')[-1])
                ones_div = 1 - ratio
                loaded = np.concatenate((np.ones((self.config.data.image_size,int(self.config.data.image_size*ones_div))), np.zeros((self.config.data.image_size,int(self.config.data.image_size*ratio)))),axis=1)
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif 'inp' in deg and 'sr' in deg:
                ratio = int(deg.split('sr')[-1])
                loaded = np.zeros((self.config.data.image_size,self.config.data.image_size)) # initialize the mask

                # nearest neighbor interpolation mask creation. Use ratio to determine where to sample pixels
                for i in range(0,loaded.shape[0],ratio):
                    for j in range(0,loaded.shape[1],ratio):
                        loaded[i,j] = 1
                
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif 'inp' in deg and deg[0] == 's':
                sflag = True
                from ddrm_codes_2.functions.svd_replacement import SuperInpainting

                ratio = int(deg.split('inp')[-1])
                loaded = np.zeros((self.config.data.image_size,self.config.data.image_size)) # initialize the mask

                # nearest neighbor interpolation mask creation. Use ratio to determine where to sample pixels
                for i in range(0,loaded.shape[0],ratio):
                    for j in range(0,loaded.shape[1],ratio):
                        loaded[i,j] = 1
                
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long() * 3
            
            if config.data.channels == 3:
                missing_g = missing_r + 1
                missing_b = missing_g + 1
                missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            else:
                missing = torch.div(missing_r, 3, rounding_mode='floor') # divide by 3 because there is only 1 image channel
            if not sflag:
                H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
            else:
                H_funcs = SuperInpainting(config.data.channels, config.data.image_size, missing, ratio, self.device)   
        elif deg == 'deno':
            from ddrm_codes_2.functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_uni':
            from ddrm_codes_2.functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from ddrm_codes_2.functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg[:2] == 'sr':
            if len(deg) <= 4:
                blur_by = int(deg[2:])
                from ddrm_codes_2.functions.svd_replacement import SuperResolution
                H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
            elif 'test3' in deg:
                blur_by = int(deg[2])
                from ddrm_codes_2.functions.svd_replacement import SuperPainting
                H_funcs = SuperPainting(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'color':
            from ddrm_codes_2.functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        else:
            print("ERROR: degradation type not supported")
            quit()
        config.args.sigma_0 = 2 * config.args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = config.args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = config.args.subset_start
        idx_so_far = config.args.subset_start
        avg_psnr = 0
        avg_psnr_bc = 0.0
        pbar = tqdm.tqdm(val_loader)

        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            if not os.path.exists(f'{out_folder}{os.sep}temp'):
                os.mkdir(f'{out_folder}{os.sep}temp')

            tvu.save_image((x_orig).cpu(),f'{out_folder}{os.sep}temp{os.sep}x_orig.png',nrow=int(self.config.sampling.batch_size**0.5))
            
            if x_orig.shape[1] != self.config.data.channels:
                x_orig = x_orig.repeat([1,3,1,1])

            y_0 = H_funcs.H(x_orig) # applies degradation matrix
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0) # add a little bit of noise to the corrupted vector
            
            # note: view is used here in the same way that reshape would be used; view is good though because it conserves memory
            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size) # multiply degraded vector by pseudo-inverse of H
            
            tvu.save_image((pinv_y_0).cpu(),f'{out_folder}{os.path.sep}temp{os.sep}pinv_y_0.png',nrow=int(self.config.sampling.batch_size**0.5))
            

            # perform bicubic interpolation and save for reference
            if deg[:2] == 'sr':
                y_scaled = inverse_data_transform(config, y_0.clip(-1,1)).view(x_orig.shape[0], x_orig.shape[1], x_orig.shape[-1]//H_funcs.ratio,x_orig.shape[-1]//H_funcs.ratio)
                bc_out = Resize(x_orig.shape[-1], interpolation=InterpolationMode.BICUBIC)(y_scaled).to(self.device)
                tvu.save_image((bc_out).cpu(),f'{out_folder}{os.path.sep}temp{os.sep}bc_out.png',nrow=int(self.config.sampling.batch_size**0.5))
            # elif deg[:3] == 'inp':
            #     N = x_orig.shape[-1]
            #     flat_mask = mask.flatten()
            #     x = np.mod(np.arange(image.size), N) # why are we doing this?
            #     y = np.arange(image.size) // N # why is this not modulus operator?
            #     mask_image = image * mask
            #     values = mask_image.flatten()
            #     interp_function = scipy.interpolate.LinearNDInterpolator(points[flat_mask], values[flat_mask])
            #     interp_image = mask_image.copy()
            #     interp_image[~mask] = interp_function(x[~flat_mask], y[~flat_mask])
            #     interp_function(x[~flat_mask], y[~flat_mask])


            
            if deg == 'deblur_uni' or deg == 'deblur_gauss': pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            elif deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1, 3, 1, 1)
            elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1 # adds some more inpainting text

            for i in range(len(pinv_y_0)):
                tvu.save_image(
                    inverse_data_transform(config, pinv_y_0[i]), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                )

            ##Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            begin = time.time()

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

            end = time.time()
            elapsed = end - begin
            print(elapsed)

            x = [inverse_data_transform(config, y) for y in x]

            

            for i in [-1]: #range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(x)-1 or i == -1:
                        orig = inverse_data_transform(config, x_orig[j])
                        mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr = avg_psnr + psnr

                        # calculate bicubic psnr for reference
                        y_scaled = inverse_data_transform(config, y_0[j].clip(-1,1)).view(1, x_orig.shape[1], x_orig.shape[-1]//H_funcs.ratio,x_orig.shape[-1]//H_funcs.ratio)
                        bc_out = Resize(orig.shape[-1], interpolation=InterpolationMode.BICUBIC)(y_scaled).to(self.device)
                        mse_bc = torch.mean((bc_out - orig)**2)
                        psnr_bc = 10 * torch.log10(1 / mse_bc)
                        avg_psnr_bc = avg_psnr_bc + psnr_bc


            

            idx_so_far += y_0.shape[0]

            temp = avg_psnr_bc / (idx_so_far - idx_init)
            temp2 = avg_psnr / (idx_so_far - idx_init)

            pbar.set_description(f'PSNR: {temp2:.2f}, PSNR-bicubic: {temp:.2f}')

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps # 1000/20 = 50 
        seq = range(0, self.num_timesteps, skip) # crete range from 0 to 1000 skipping 50 at each step
        out_folder = os.path.join(self.config.args.image_folder,'temp')

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes, out_folder=out_folder)
        if last:
            x = x[0][-1]
        return x

