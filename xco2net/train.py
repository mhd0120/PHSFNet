from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import os
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .process_data import *
from .support_func import r2


SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


def _split_model_output(output):
    if isinstance(output, tuple):
        return output[0], output[1] if len(output) > 1 else {}
    return output, {}


def _forward_model(model, features, radiances, return_aux=False):
    if return_aux:
        try:
            return _split_model_output(model(features, radiances, return_aux=True))
        except TypeError:
            return _split_model_output(model(features, radiances))
    return _split_model_output(model(features, radiances))


def _gate_mean_from_aux(aux):
    gates = aux.get("band_gates") if isinstance(aux, dict) else None
    if gates is None:
        return None
    return gates.detach().cpu().mean(dim=0).numpy()
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

def my_plot(x, y, fig, ax):
    LBL = x[:10000] * 1e6  # 让数值回归到正常范围
    MLP = y[:10000] * 1e6

    xy = np.vstack([MLP, LBL])
    z = gaussian_kde(xy)(xy)
    z=(z-np.min(z))/(np.max(z)-np.min(z))
    idx = z.argsort()
    MMLP, LLBL, z = MLP[idx], LBL[idx], z[idx]
    
    # min_value = 0.99*np.min(LBL)
    # max_value = 1.01*np.max(LBL)
    min_value = 380
    max_value = 430
    xyline = np.linspace(min_value,max_value,61)
    

    # fig,ax = plt.subplots(figsize=(4,3))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax.set_ylabel('Predicted [ppm]')
    ax.set_xlabel('Ground truth [ppm]')
    ax.axis([min_value,max_value,min_value,max_value])
    ax.set_aspect(1.0)
    text1 = 'N:'+str(len(LBL)) 
    R2_test = r2(LBL,MLP)
    text2 = 'R$^2$:'+str("%.3f" % R2_test)
    text3 = 'MAE: '+str("%.3f" % mean_absolute_error(LBL, MLP))+' ppm'
    text4 = 'RMSE: '+str("%.3f" % np.sqrt(mean_squared_error(LBL, MLP)))+' ppm'
    ax.plot(xyline, xyline, 'r-', label=r'Ideal$\pm$1%', linewidth=2)
    ax.fill_between(xyline, xyline*1.01, xyline*0.99,
        alpha=0.5, edgecolor='0.4', facecolor='0.4',
        linewidth=1, linestyle='--', antialiased=True)
    cf = ax.scatter(LLBL, MMLP, c=z,s=3, alpha=1.0)#facecolors='none',
    fig.colorbar(cf, label='Number density')

    ax.set_xlim([380, 430])
    ax.set_ylim([380, 430])

    z0 = 0.97
    dz = 0.07
    ax.text(0.02,z0,text1,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    ax.text(0.02,z0-dz,text2,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    ax.text(0.02,z0-dz*2,text3,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    ax.text(0.02,z0-dz*3,text4,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    return fig, ax

def predict(data, model, device, batch_size, stad_info, feature_scaler):
    model.eval()
    with torch.no_grad():
        x_all, y_all = [], []
        label_mean = stad_info[0]
        label_std = stad_info[1]
        
        for data in DataLoader(data, batch_size=batch_size):
            features = data["features"].to(device)
            radiances = data["spectra"].to(device)
            labels = data["labels"]
            output, _ = _forward_model(model, features, radiances)
            
            xco2_priori = feature_scaler.inverse_transform(features.cpu().numpy())[:, -1]
            x_all.append(labels.numpy().flatten() * float(label_std) + float(label_mean) + xco2_priori)  # label+xco2先验
            y_all.append(output.cpu().numpy().flatten() * float(label_std) + float(label_mean) + xco2_priori)  # output+xco2先验
    return np.concatenate(x_all), np.concatenate(y_all)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        return False

class TrainInClass:
    def __init__(self, name: str, file_name: str, model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 epochs: int, batch_size: int,
                 early_stopper: EarlyStopper, device: torch.device,
                 datas: List[torch.utils.data.Dataset], stad_info: List, feature_scaler):
        self.writer = SummaryWriter(name)
        self.file_name = file_name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopper = early_stopper
        self.device = device
        self.datasets = datas
        self.stad_info = stad_info
        self.feature_scaler = feature_scaler
        self.best_valid_loss = float('inf')  # Track the best validation loss
        self.output_dir = os.path.join("outputs", self.file_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.gate_statistics_path = os.path.join(self.output_dir, "gate_statistics.csv")
        self.test_gate_values_path = os.path.join(self.output_dir, "test_gate_values.csv")
        self.test_gate_summary_path = os.path.join(self.output_dir, "test_gate_summary.csv")
        for path in [self.gate_statistics_path, self.test_gate_values_path, self.test_gate_summary_path]:
            if os.path.exists(path):
                os.remove(path)

    def make_plot(self, plot_max_n = 1000):
        """_summary_

        Parameters
        ----------
        model : _type_
            _description_
        datas : _type_
            datas = [train_data, valid_data, test_data]  # 更新为三个数据集
        plot_max_n : int, optional
            _description_, by default 500

        Returns
        -------
        _type_
            _description_
        """
        fig, axs = plt.subplots(1,3, figsize=[12, 3])

        
        titles = ["Train", "Valid", "Test"]  # 移除了"Future"
        for i, dataset in enumerate(self.datasets):
            ax = axs[i]
            n = len(dataset)
            n = min(plot_max_n, n)
            subset = random_subset(dataset, n)
            x, y = predict(subset, self.model, self.device, self.batch_size, 
                           self.stad_info, self.feature_scaler)
            fig, _ = my_plot(x, y, fig, ax)  # 画图
            ax.set_title(titles[i])
        fig.tight_layout()
        return fig

    def _append_gate_statistics(self, epoch, gate_mean):
        file_exists = os.path.exists(self.gate_statistics_path)
        with open(self.gate_statistics_path, "a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["epoch", "mean_gate_o2", "mean_gate_wco2", "mean_gate_sco2"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch + 1,
                "mean_gate_o2": float(gate_mean[0]),
                "mean_gate_wco2": float(gate_mean[1]),
                "mean_gate_sco2": float(gate_mean[2]),
            })

    def _export_best_test_gates(self):
        test_dataset = self.datasets[2]
        best_weight_path = f"model_status/{self.file_name}/weight/best.pth"
        if not os.path.exists(best_weight_path):
            return

        self.model.load_state_dict(torch.load(best_weight_path, map_location=self.device))
        self.model.eval()

        label_mean = float(self.stad_info[0])
        label_std = float(self.stad_info[1])
        rows = []
        gate_values = []
        sample_offset = 0
        with torch.no_grad():
            loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            for data in loader:
                features = data["features"].to(self.device)
                radiances = data["spectra"].to(self.device)
                labels = data["labels"]
                output, aux = _forward_model(self.model, features, radiances, return_aux=True)
                gates = aux.get("band_gates") if isinstance(aux, dict) else None
                if gates is None:
                    return

                xco2_priori = self.feature_scaler.inverse_transform(features.cpu().numpy())[:, -1]
                y_true = labels.numpy().flatten() * label_std + label_mean + xco2_priori
                y_pred = output.cpu().numpy().flatten() * label_std + label_mean + xco2_priori
                gates_np = gates.detach().cpu().numpy()
                gate_values.append(gates_np)

                for i in range(len(y_true)):
                    rows.append({
                        "sample_index": sample_offset + i,
                        "gate_o2": float(gates_np[i, 0]),
                        "gate_wco2": float(gates_np[i, 1]),
                        "gate_sco2": float(gates_np[i, 2]),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                        "residual": float(y_pred[i] - y_true[i]),
                    })
                sample_offset += len(y_true)

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.test_gate_values_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "sample_index", "gate_o2", "gate_wco2", "gate_sco2",
                    "y_true", "y_pred", "residual",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        gates_all = np.concatenate(gate_values, axis=0)
        with open(self.test_gate_summary_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "test_gate_o2_mean", "test_gate_o2_std",
                    "test_gate_wco2_mean", "test_gate_wco2_std",
                    "test_gate_sco2_mean", "test_gate_sco2_std",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "test_gate_o2_mean": float(gates_all[:, 0].mean()),
                "test_gate_o2_std": float(gates_all[:, 0].std()),
                "test_gate_wco2_mean": float(gates_all[:, 1].mean()),
                "test_gate_wco2_std": float(gates_all[:, 1].std()),
                "test_gate_sco2_mean": float(gates_all[:, 2].mean()),
                "test_gate_sco2_std": float(gates_all[:, 2].std()),
            })
        print(f"Saved gate statistics to {self.output_dir}")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            # 初始化训练损失（带权重）
            train_loss = 0.0
            total_samples = 0  # 总样本数（用于加权平均）

            train_dataset, valid_dataset, _, = self.datasets

            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=0)  #num_workers=5 Linux用5，win用0

            # 移除了tqdm进度条
            for data in train_dataloader:
                features = data["features"].to(self.device)
                radiances = data["spectra"].to(self.device)
                labels = data["labels"].to(self.device)
                error_weights = data["error_weights"].to(self.device)
                self.optimizer.zero_grad()

                output, _ = _forward_model(self.model, features, radiances)
                loss = self.criterion(output, labels)
                # 使用error_weights加权计算batch损失
                weighted_loss = (loss * error_weights).mean()

                weighted_loss.backward()
                self.optimizer.step()

                # 计算加权训练损失
                batch_samples = features.size(0)
                train_loss += weighted_loss.item() * batch_samples
                total_samples += batch_samples

            # 计算本epoch平均训练损失
            train_loss_avg = train_loss / total_samples
            # 将训练损失写入TensorBoard
            self.writer.add_scalars('Loss', {"train": train_loss_avg}, epoch)

            # 验证阶段
            self.model.eval()
            valid_loss = 0.0
            total_valid_samples = 0
            valid_gate_sum = None
            valid_gate_count = 0
            with torch.no_grad():
                valid_dataloader = DataLoader(
                    valid_dataset, batch_size=self.batch_size, shuffle=True,
                    num_workers=0)  # num_workers=5
                for data in valid_dataloader:
                    features = data["features"].to(self.device)
                    radiances = data["spectra"].to(self.device)
                    labels = data["labels"].to(self.device)
                    error_weights = data["error_weights"].to(self.device)

                    output, aux = _forward_model(self.model, features, radiances, return_aux=True)
                    loss = self.criterion(output, labels)
                    gates = aux.get("band_gates") if isinstance(aux, dict) else None
                    if gates is not None:
                        gate_batch = gates.detach().cpu()
                        gate_sum = gate_batch.sum(dim=0)
                        valid_gate_sum = gate_sum if valid_gate_sum is None else valid_gate_sum + gate_sum
                        valid_gate_count += gate_batch.size(0)
                    # 计算加权验证损失
                    batch_valid_samples = features.size(0)
                    valid_loss += (loss * error_weights).mean().item() * batch_valid_samples
                    total_valid_samples += batch_valid_samples
                valid_loss /= total_valid_samples
            valid_gate_mean = None
            if valid_gate_sum is not None and valid_gate_count > 0:
                valid_gate_mean = (valid_gate_sum / valid_gate_count).numpy()

            # 将验证损失写入TensorBoard
            self.writer.add_scalars('Loss', {"valid": valid_loss}, epoch)
            if valid_gate_mean is not None:
                self.writer.add_scalars(
                    'BandGates/valid_mean',
                    {
                        "o2": float(valid_gate_mean[0]),
                        "weak_co2": float(valid_gate_mean[1]),
                        "strong_co2": float(valid_gate_mean[2]),
                    },
                    epoch,
                )
                self._append_gate_statistics(epoch, valid_gate_mean)

            # 打印本epoch的训练/验证损失
            print(f"Epoch [{epoch + 1}/{self.epochs}] - "
                  f"Train Loss: {train_loss_avg:.5e}, "
                  f"Valid Loss: {valid_loss:.5e}"
                  )
            if valid_gate_mean is not None:
                print(
                    "Valid gates: "
                    f"O2={valid_gate_mean[0]:.4f}, "
                    f"weak CO2={valid_gate_mean[1]:.4f}, "
                    f"strong CO2={valid_gate_mean[2]:.4f}"
                )

            # 每5个epoch生成一次图表
            # if epoch % 5 == 0:
            #     if epoch % 40 == 20:
            #         fig = self.make_plot(100000)
            #     else:
            #         fig = self.make_plot()
            #     fig_name = f"model_status/{self.file_name}/img/{epoch:03d}.png"
            #     fig.savefig(fig_name)
            #     self.writer.add_figure('Scatter', fig, epoch)

            # 早停检查
            if self.early_stopper.early_stop(valid_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # 保存模型权重 - 只保存验证损失最好的模型
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f"model_status/{self.file_name}/weight/best.pth")
                print(f"Saved best model at epoch {epoch + 1} with valid loss: {valid_loss:.5e}")

            # 更新学习率
            self.scheduler.step()

        self._export_best_test_gates()

