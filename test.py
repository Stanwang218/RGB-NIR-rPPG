import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as sio
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils.model.model_disentangle import HR_disentangle_cross;
from utils.loss.loss_cross import Cross_loss;
from utils.loss.loss_r import Neg_Pearson;
from utils.loss.loss_SNR import SNR_loss;
from tqdm import tqdm
import argparse
import yaml
from utils.dataset.mrnirp_dataset import MSTmap_dataset_cut
# from utils.dataset.pure_dataset import MSTmap_PURE_cut
from utils.dataset.sig_util import compute_metric_per_clip
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import random
import scipy
from tqdm import tqdm
from utils.model.models_vit import vit_base_patch16
from utils.model.models_mae import mae_vit_base_patch16_dec512d8b
import matplotlib.pyplot as plt

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
        
    return SNR

def set_seed(seed):
    if seed != 0:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def get_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument("--dataset", required=False, default='/mae_rppg/train/fold1.yaml')
    parser.add_argument("--runner", required=False, default='/mae_rppg/train/fold1.yaml')
    parser.add_argument("--model", required=False, default='/mae_rppg/train/fold1.yaml')
    parser.add_argument("--seed", type=int, default=7234)
    parser.add_argument("--selected_topics", nargs = '+', type=str, default=[])
    return parser

def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr-HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / HR_rel) * 100
    p = np.sum((HR_pr - np.mean(HR_pr))*(HR_rel - np.mean(HR_rel))) / (
                0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mape: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p

def train(net, model_name = 'vit'):
    global min_val_loss
    global cur_patience
    net.train()
    train_loss = 0
    fps = torch.tensor([30])
    
    for batch_idx, (data, bvp, bpm, name) in tqdm(enumerate(train_loader)):

        data = Variable(data)
        bvp = Variable(bvp)
        bpm = Variable(bpm.view(-1,1))
        fps = Variable(fps.view(-1,1))
        data, bpm = data.cuda(), bpm.cuda()
        fps = fps.cuda()
        bvp = bvp.cuda()

        feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data)

        loss_hr = lossfunc_HR(output, bpm)*lambda_hr; # predicted heart rate against ground truth
        loss_img = lossfunc_img(data, img_out)*lambda_img; # reconstruction loss
        loss_ecg = lossfunc_ecg(ecg, bvp)*lambda_ecg; # rPPG loss

        # print(loss_ecg)
        loss_SNR, tmp = lossfunc_SNR(ecg, bpm, fps, pred = output, flag = None);
        loss_SNR = loss_SNR * lambda_snr
        loss = loss_hr + loss_ecg + loss_img + loss_SNR;

        loss_cross, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, loss_hr_dis1, loss_hr_dis2 = lossfunc_cross(feat_hr, feat_n, output,
                                                                                                                            feat_hrf1, feat_nf1,
                                                                                                                            hrf1, idx1,
                                                                                                                            feat_hrf2, feat_nf2,
                                                                                                                            hrf2, idx2, bpm)
        
        # print(output.view(-1))
        loss = loss + loss_cross;

        train_loss += loss.item();

        optimizer.zero_grad()
        loss.backward()
        optimizer.step();

        print('Train epoch: {:.0f}, it: {:.0f}, loss: {:.4f}, loss_hr: {:.4f}, loss_img: {:.4f}, loss_cross: {:.4f}, loss_snr: {:.4f}, loss_ecg: {:.4f}'.format(epoch, batch_idx,
                                                                                    loss, loss_hr, loss_img, loss_cross, loss_SNR, loss_ecg));
    # valid
    val_loss = 0
    with torch.no_grad():
        for data, bvp, bpm in valid_loader:
            data = Variable(data);
            bvp = Variable(bvp);
            bpm = Variable(bpm.view(-1,1));
            fps = Variable(fps.view(-1,1));
            data, bpm = data.cuda(), bpm.cuda();
            fps = fps.cuda()
            bvp = bvp.cuda()
            feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data);
            loss_hr = lossfunc_HR(output, bpm)*lambda_hr; # predicted heart rate against ground truth
            loss_img = lossfunc_img(data, img_out)*lambda_img; # reconstruction loss
            loss_ecg = lossfunc_ecg(ecg, bvp)*lambda_ecg; # rPPG loss
            loss_SNR, tmp = lossfunc_SNR(ecg, bpm, fps, pred = output, flag = None)
            loss_SNR = loss_SNR * lambda_snr
            loss = loss_hr + loss_ecg + loss_img + loss_SNR;

            loss_cross, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, loss_hr_dis1, loss_hr_dis2 = lossfunc_cross(feat_hr, feat_n, output,
                                                                                                                                feat_hrf1, feat_nf1,
                                                                                                                                hrf1, idx1,
                                                                                                                                feat_hrf2, feat_nf2,
                                                                                                                                hrf2, idx2, bpm)
            loss = loss + loss_cross;

            val_loss += loss.item();
    
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(valid_loader)
    print(f"Current loss {val_loss}")
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        model_ckpt = net.state_dict()
        print(f"best loss {val_loss}")
        if not os.path.exists(runner_config['model_saved_path']):
            os.mkdir(runner_config['model_saved_path'])
        torch.save(model_ckpt, f"{runner_config['model_saved_path']}/best_model_{task_name}.pt")
        print(f"{runner_config['model_saved_path']}/best_model_{task_name}.pt")
        cur_patience = 0
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    cur_patience += 1
        
def test():
    net.eval()
    test_loss = 0;
    true_hr, pred_hr = [], []
    snr_all = []
    total_time = 0
    with torch.no_grad():
        for (data, bvp, hr, name) in test_loader:
            # print(hr)
            # hr = hr * 60 * 30 / 224
            # print(hr)
            true_hr.extend(hr.tolist())
            data = Variable(data);
            hr = Variable(hr.view(-1,1));
            data, hr = data.cuda(), hr.cuda();
            
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data);
            
            
            ender.record()
            torch.cuda.synchronize()
            total_time += starter.elapsed_time(ender)/1000
            
            loss = lossfunc_HR(output, hr);
            output = output.squeeze() # * 60 * 30 / 224
            pred_hr.extend(output.tolist())
            snr_all.append(ecg.cpu().numpy())
            test_loss += loss.item();
            
    pred_hr = [round(hr, 4) for hr in pred_hr]
    snr_all = np.concatenate(snr_all, axis = 0)
    snr_list = []
    for snr, hr in zip(snr_all, pred_hr):
        snr_list.append(_calculate_SNR(snr, hr))
    MyEval(pred_hr, true_hr)
    print(f"snr:{np.array(snr_list).mean()}")
    if task == 'test':
        print(pred_hr)
        print(true_hr)
    print('Throughput:', len(test_dataset) * 224 / total_time, 'FPS')


def train_vit(runner_config, model, train_loader, val_loader, task = 'finetune'):
    if task == 'finetune':
        checkpoint_model = torch.load(runner_config['pretrain_path'], map_location='cpu')
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        
    epoch_num = runner_config['epochs']
    learning_rate = runner_config['lr']
    patience, cur_patience  = runner_config['patience'], 0

    device = runner_config['device']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay = 0.05)
    
    model.to(device)
    
    lossfunc_ecg = Neg_Pearson(downsample_mode = 0)
    
    train_loss_list, val_loss_list = [], []
    
    min_val_loss = torch.finfo(torch.float32).max
    for epoch in tqdm(range(epoch_num)):
        temp_tr_loss, temp_val_loss = 0, 0
        for batch_idx, (data, bvp, bpm, name) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            bvp = bvp.to(device)
            pred = model(data)
            optimizer.zero_grad()
            loss = lossfunc_ecg(bvp, pred)
            loss.backward()
            optimizer.step()
            temp_tr_loss += loss.item()
        train_loss_list.append(temp_tr_loss / len(train_loader))
        for batch_idx, (data, bvp, bpm, name) in tqdm(enumerate(val_loader)):
            data = data.to(device)
            bvp = bvp.to(device)
            pred = model(data)
            loss = lossfunc_ecg(bvp, pred)
            temp_val_loss += loss.item()
        if min_val_loss > temp_val_loss:
            min_val_loss = temp_val_loss
            cur_patience = 0
            model_ckpt = model.state_dict()
            print(f"best loss {min_val_loss / len(val_loader)}")
            if not os.path.exists(runner_config['model_saved_path']):
                os.mkdir(runner_config['model_saved_path'])
            torch.save(model_ckpt, f"{runner_config['model_saved_path']}/ckpt.pth")
        else:
            cur_patience += 1
            
        val_loss_list.append(temp_val_loss / len(val_loader))
        if cur_patience >= patience:
            print("no more patience, stop training")
            break
    
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label = 'val')
    plt.legend()
    plt.savefig(f"{runner_config['log']}/loss.png")

def test_vit(runner_config, model:nn.Module, test_loader):
    ckpt_path = f"{runner_config['model_saved_path']}/ckpt.pth"
    device = runner_config['device']
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(device)
    pred_bvp_list = []
    bpm_list = []
    with torch.no_grad():
        for (batch_idx, (data, bvp, bpm, name)) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            pred_bvp = model(data).cpu() # bz, 224 
            bpm_list.extend(bpm.cpu().tolist())
            pred_bvp_list.append(pred_bvp.cpu().numpy())
            break
    
    pred_bvp_list = np.vstack(pred_bvp_list)
    # extract heart rate from bvp
    
    hr_pred = [compute_metric_per_clip(pred_bvp_list[i, :]) for i in range(pred_bvp_list.shape[0])]


def train_mae(runner_config, model, train_loader, val_loader):
    epoch_num = runner_config['epochs']
    learning_rate = runner_config['lr']
    patience, cur_patience  = runner_config['patience'], 0

    device = runner_config['device']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay = 0.05)
    
    model.to(device)
    
    train_loss_list, val_loss_list = [], []
    
    min_val_loss = torch.finfo(torch.float32).max
    for epoch in tqdm(range(epoch_num)):
        temp_tr_loss, temp_val_loss = 0, 0
        for batch_idx, (data, bvp, bpm, name) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            loss, bvp_map, mask = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_tr_loss += loss.item()
        train_loss_list.append(temp_tr_loss / len(train_loader))
        for batch_idx, (data, bvp, bpm, name) in tqdm(enumerate(val_loader)):
            data = data.to(device)
            loss, bvp_map, mask = model(data)
            temp_val_loss += loss.item()
        if min_val_loss > temp_val_loss:
            min_val_loss = temp_val_loss
            cur_patience = 0
            model_ckpt = model.state_dict()
            print(f"best loss {min_val_loss}")
            if not os.path.exists(runner_config['model_saved_path']):
                os.mkdir(runner_config['model_saved_path'])
            torch.save(model_ckpt, f"{runner_config['model_saved_path']}/ckpt.pth")
        else:
            cur_patience += 1
            
        val_loss_list.append(temp_val_loss / len(val_loader))
        if cur_patience >= patience:
            print("no more patience, stop training")
            break
    
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label = 'val')
    plt.legend()
    plt.savefig(runner_config['log'])
    
def test_mae(runner_config, model:nn.Module, test_loader):
    ckpt_path = f"{runner_config['model_saved_path']}/ckpt.pth"
    device = runner_config['device']
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    bvp_map_list = []
    with torch.no_grad():
        for (batch_idx, (data, bvp, bpm, name)) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            loss, bvp_map, mask = model(data)
            bvp_map_list.append(bvp_map.cpu().numpy())
            break
    bvp_map_list = np.vstack(bvp_map_list)
    np.save(f"{runner_config['model_saved_path']}/example.npy")
            


if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()
    set_seed(args.seed)
    f_dataset, f_runner, f_model = open(args.dataset), open(args.runner), open(args.model)
    dataset_config, model_config, runner_config = yaml.safe_load(f_dataset), yaml.safe_load(f_model), yaml.safe_load(f_runner)

    dataset_config['selected_topic'] = args.selected_topics
    print(dataset_config['selected_topic'])
    saved_dir = os.path.join(runner_config['root'], runner_config['name'] + "_" + model_config['name'])

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    runner_config['model_saved_path'] = os.path.join(saved_dir, 'ckpt')
    runner_config['log'] = saved_dir
        
    task = runner_config['task']
    task_name = runner_config['name']

    train_dataset, test_dataset, valid_dataset = MSTmap_dataset_cut.split_dataset(config=dataset_config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=runner_config['batch_size'],)
    test_loader = DataLoader(dataset=test_dataset, batch_size=runner_config['batch_size'],)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=runner_config['batch_size'],)
    
    if model_config['name'] == 'vit':
        model = vit_base_patch16(in_chans = 3, num_classes = 224)
        train_vit(runner_config, model, train_loader, valid_loader)
        test_vit(runner_config, model, test_loader)
    elif model_config['name'] == 'mae':
        model = mae_vit_base_patch16_dec512d8b(in_chans=3, decoder_embed_dim=128, decoder_depth=8)
        train_mae(runner_config, model, train_loader, valid_loader)
        test_mae(runner_config, model, test_loader)
    else:
        exit()

    # lambda_hr = 1; # correct
    # lambda_img = 50; # 50
    # lambda_ecg = 20; # 2
    # lambda_snr = 10; # correct
    # lambda_cross_fhr = 10; # 10
    # lambda_cross_fn = 10; # 10
    # lambda_cross_hr = 1; # correct
    # # old, 1,50,20,10
    # # new, 1, 5, 10, 1

    # print(f"lambda hr: {lambda_hr}, lambda img: {lambda_img}, lambda ecg: {lambda_ecg}, lambda snr: {lambda_snr}, lr: {learning_rate}")

    # video_length = 224;
    # net = HR_disentangle_cross(video_length, num_channel=model_config['channels']);
    # if model_config['pretrained_path'] is not None:
    #     ckpt_path = model_config['pretrained_path']
    #     net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    #     print(f"successfully load ckpt from {ckpt_path}")
        
    # net.cuda();
    # #########################################################################

    # lossfunc_HR = nn.L1Loss();
    # # lossfunc_HR = nn.MSELoss()
    # lossfunc_img = nn.L1Loss();
    # lossfunc_cross = Cross_loss(lambda_cross_fhr = lambda_cross_fhr, lambda_cross_fn = lambda_cross_fn, lambda_cross_hr = lambda_cross_hr);
    # lossfunc_ecg = Neg_Pearson(downsample_mode = 0)
    # lossfunc_SNR = SNR_loss(clip_length = video_length, loss_type = 7);

    # scheduler = MultiStepLR(optimizer, milestones=[15,25], gamma=0.5)
    # min_val_loss = 90000

    # train_loss_list, val_loss_list = [], []

    # begin_epoch = 1;

    # if task == 'train':
    #     for epoch in range(begin_epoch, epoch_num + 1):
    #         train();
    #         test();
    #         scheduler.step()
    #         if cur_patience >= patience:
    #             print("no more patience, stop training")
    #             break

    #     plt.plot([i + 1 for i in range(len(train_loss_list))], train_loss_list, label='Train')
    #     plt.plot([i + 1 for i in range(len(val_loss_list))], val_loss_list, label='Val')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     if not os.path.exists(runner_config['log']):
    #         os.mkdir(runner_config['log'])
    #     plt.savefig(f"{runner_config['log']}/loss_{task_name}.png")

    # else:
    #     print("start testing")
    #     test()