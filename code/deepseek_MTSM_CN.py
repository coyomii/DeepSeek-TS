import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv(r"data\sales_data.csv")

# df.plot(x="date", y=["product1_sales", "product2_sales", "product3_sales", "product4_sales", "product5_sales"], figsize=(12, 6))
# plt.title("Simulated Sales Data for 5 Products")
# plt.show()

# -------------------------------
#  数据准备：带标准化的时间序列预测数据集
# -------------------------------
class SalesDataset(Dataset):
    def __init__(self, df, input_window=30, forecast_horizon=5):
        """
        df: DataFrame，包含列：date, product1_sales, ..., product5_sales
        input_window: 用作输入的天数
        forecast_horizon: 预测天数；目标是这些天内每个产品的平均销售额
        """
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        self.df = df.reset_index(drop=True)
        # 仅使用销售列（除了'date'以外的所有列）
        data = df.drop(columns=['date']).values.astype(np.float32)
        
        # 计算整个数据集的标准化参数
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-6  # 避免除以零
        
        # 计算整个数据集的标准化参数
        self.data = (data - self.mean) / self.std
        self.n_samples = len(self.data) - input_window - forecast_horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 输入：input_window天的销售数据（标准化后）
        x = self.data[idx: idx + self.input_window]
        # 目标：未来forecast_horizon天的平均销售额（标准化后）
        y = np.mean(self.data[idx + self.input_window: idx + self.input_window + self.forecast_horizon], axis=0)
        return x, y

def prepare_dataloaders(df, input_window=30, forecast_horizon=5, batch_size=32, train_ratio=0.8):
    """
    准备训练和验证数据加载器
    """
    dataset = SalesDataset(df, input_window, forecast_horizon)
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    # 时序切分：前n_train个样本用于训练，剩余用于验证
    train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(n_train, n_total)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = prepare_dataloaders(df, input_window=30, forecast_horizon=5, batch_size=32, train_ratio=0.8)

# -------------------------------
#  基础GRU预测模型
# -------------------------------
class SimpleGRUForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2):
        super(SimpleGRUForecastingModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc_forecast = nn.Linear(hidden_dim, num_products)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        forecast = self.fc_forecast(last_hidden)
        return forecast

# -------------------------------
#  基于GRPO的预测模型
# -------------------------------
class ForecastingGRPOModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2, lambda_policy=0.1):
        """
        该模型使用GRU编码器预测未来forecast_horizon天内每个产品的平均销售额
        包含一个策略分支来计算GRPO启发的损失
        """
        super(ForecastingGRPOModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc_forecast = nn.Linear(hidden_dim, num_products)
        self.policy_net = nn.Linear(hidden_dim, 1)
        self.lambda_policy = lambda_policy

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim)
        forecast = self.fc_forecast(last_hidden)  # (batch, num_products)
        policy_value = self.policy_net(last_hidden)  # (batch, 1)
        return forecast, policy_value

# -------------------------------
#  带扩展MLA机制的GRPO预测模型
# -------------------------------
class ForecastingGRPOMLAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.3, lambda_policy=0.06):
        """
        该模型通过引入扩展的MLA(Mamba风格)机制来扩展GRPO预测方法
        潜在状态以状态空间方式更新，对整个更新应用非线性激活
        """
        super(ForecastingGRPOMLAModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lambda_policy = lambda_policy
        self.dropout = nn.Dropout(dropout)
        # 将输入映射到潜在空间
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        # 状态空间转移矩阵(M)
        self.M = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 完整状态更新的非线性激活函数
        self.activation = nn.ReLU()
        self.fc_forecast = nn.Linear(hidden_dim, num_products)
        self.policy_net = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # 初始化潜在状态为零
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        # 迭代更新潜在状态
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            # 首先计算不带激活的修正
            correction = self.input_transform(x_t)
            # 使用ReLU激活更新潜在状态
            h = self.activation(self.M(h) + correction)
            h = self.dropout(h)
        forecast = self.fc_forecast(h)
        policy_value = self.policy_net(h)
        return forecast, policy_value

# -------------------------------
#  辅助函数：ARMA预测
# -------------------------------
def arma_forecast(series, forecast_horizon):
    """
    对提供的序列拟合ARIMA(1,0,1)模型并预测forecast_horizon步
    返回平均预测值
    """
    try:
        arma_model = ARIMA(series, order=(1, 0, 1))
        arma_fit = arma_model.fit(disp=0)
        forecast = arma_fit.forecast(steps=forecast_horizon)
        return np.mean(forecast)
    except Exception as e:
        return series[-1]

def evaluate_arma(df, input_window=30, forecast_horizon=5, train_ratio=0.8):
    """
    对每个产品（列）在验证期间使用滚动ARMA预测
    返回每个产品的MAPE值字典和总体平均MAPE
    """
    n = len(df)
    train_end = int(n * train_ratio)
    products = [col for col in df.columns if col != "date"]
    mape_dict = {}
    all_mapes = []
    
    for prod in products:
        preds = []
        actuals = []
        for i in range(train_end - input_window, n - input_window - forecast_horizon + 1):
            series = df[prod].values[:i + input_window]
            pred = arma_forecast(series, forecast_horizon)
            preds.append(pred)
            actual = np.mean(df[prod].values[i + input_window: i + input_window + forecast_horizon])
            actuals.append(actual)
        preds = np.array(preds)
        actuals = np.array(actuals)
        prod_mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-6))) * 100
        mape_dict[prod] = prod_mape
        all_mapes.append(prod_mape)
    overall_mape = np.mean(all_mapes)
    return mape_dict, overall_mape

# -------------------------------
#  训练和验证函数
# -------------------------------
def train_model_full(model, dataloader, optimizer, device, grad_clip=1.0):
    """
    训练带GRPO机制的模型
    """
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        forecast, policy_value = model(x)
        # 预测损失：结合MSE和MAE
        loss_mse = F.mse_loss(forecast, y)
        loss_mae = F.l1_loss(forecast, y)
        loss_forecast = 0.5 * loss_mse + 0.5 * loss_mae
        
        # GRPO启发的策略损失
        advantage = (y - forecast).mean(dim=1, keepdim=True)
        baseline = 0.5  # 选定的常数基线
        r_t = policy_value / baseline
        epsilon = 0.1
        r_t_clipped = torch.clamp(r_t, 1 - epsilon, 1 + epsilon)
        policy_loss = -torch.min(r_t * advantage, r_t_clipped * advantage).mean()
        
        loss = loss_forecast + model.lambda_policy * policy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def train_simple_model(model, dataloader, optimizer, device, grad_clip=1.0):
    """
    训练简单GRU模型
    """
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        forecast = model(x)
        loss_mse = F.mse_loss(forecast, y)
        loss_mae = F.l1_loss(forecast, y)
        loss = 0.5 * loss_mse + 0.5 * loss_mae
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model(model, dataloader, device, dataset_obj, debug=False):
    """
    验证带GRPO机制的模型
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # 反标准化：forecast_orig = forecast * std + mean
    mean = dataset_obj.mean
    std = dataset_obj.std
    all_preds_orig = all_preds * std + mean
    all_targets_orig = all_targets * std + mean
    if debug:
        print("Prediction range:", np.min(all_preds_orig), np.max(all_preds_orig))
        print("Target range:", np.min(all_targets_orig), np.max(all_targets_orig))
    mape = np.mean(np.abs((all_targets_orig - all_preds_orig) / (all_targets_orig + 1e-6))) * 100
    return mape

def validate_simple_model(model, dataloader, device, dataset_obj, debug=False):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            forecast = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mean = dataset_obj.mean
    std = dataset_obj.std
    all_preds_orig = all_preds * std + mean
    all_targets_orig = all_targets * std + mean
    if debug:
        print("Simple GRU Prediction range:", np.min(all_preds_orig), np.max(all_preds_orig))
        print("Simple GRU Target range:", np.min(all_targets_orig), np.max(all_targets_orig))
    mape = np.mean(np.abs((all_targets_orig - all_preds_orig) / (all_targets_orig + 1e-6))) * 100
    return mape


# -------------------------------
# Main Training
# -------------------------------
def main():
    # 读取生成的销售数据
    df = pd.read_csv(r"data\sales_data.csv")
    
    # 准备按时间顺序切分的数据加载器（前80%用于训练，剩余部分用于验证）
    train_loader, val_loader = prepare_dataloaders(df, input_window=30, forecast_horizon=5, batch_size=32, train_ratio=0.8)
    
    # 获取标准化参数，以便在验证时反标准化数据
    dataset_obj = SalesDataset(df, input_window=30, forecast_horizon=5)
    
    # 选择设备：有GPU使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义模型参数
    input_dim = train_loader.dataset[0][0].shape[-1]  # 比如：5个产品
    hidden_dim = 256      # GRU的隐藏维度
    num_products = input_dim  # 预测每个产品的平均销售额
    forecast_horizon = 10  # 注意：此参数在模型中使用，尽管目标是预测未来5天
    lambda_policy = 0.06   # GRPO启发式策略损失的权重
    
    # -------------------------------
    # 训练GRPO启发的预测模型
    # -------------------------------
    model = ForecastingGRPOModel(input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2, lambda_policy=lambda_policy)
    model.to(device)
    
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    n_epochs = 22

    print("正在训练GRPO启发的预测模型...")
    # 训练GRPO模型
    for epoch in range(n_epochs):
        train_loss = train_model_full(model, train_loader, optimizer, device, grad_clip=1.0)
        mape = validate_model(model, val_loader, device, dataset_obj, debug=True)
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs} - GRPO模型训练损失: {train_loss:.4f}, MAPE: {mape:.2f}%")
    
    # -------------------------------
    # 使用ARMA进行预测并进行比较
    # -------------------------------
    print("\n正在评估ARMA模型的预测效果...")
    arma_mapes, overall_arma_mape = evaluate_arma(df, input_window=30, forecast_horizon=5, train_ratio=0.8)
    print("每个产品的ARMA MAPE:", arma_mapes)
    print("整体ARMA MAPE:", overall_arma_mape, "%")
    
    # -------------------------------
    # 训练简单GRU预测模型进行对比
    # -------------------------------
    print("\n正在训练简单的GRU预测模型进行对比...")
    simple_model = SimpleGRUForecastingModel(input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2)
    simple_model.to(device)
    optimizer_simple = torch.optim.Adam(simple_model.parameters(), lr=0.0003)
    scheduler_simple = torch.optim.lr_scheduler.StepLR(optimizer_simple, step_size=10, gamma=0.9)
    n_epochs_simple = 22
    for epoch in range(n_epochs_simple):
        train_loss_simple = train_simple_model(simple_model, train_loader, optimizer_simple, device, grad_clip=1.0)
        simple_mape = validate_simple_model(simple_model, val_loader, device, dataset_obj, debug=True)
        scheduler_simple.step()
        print(f"Epoch {epoch+1}/{n_epochs_simple} - 简单GRU模型训练损失: {train_loss_simple:.4f}, MAPE: {simple_mape:.2f}%")
    
    # -------------------------------
    # 训练带扩展MLA机制的GRPO模型
    # -------------------------------
    print("\n正在训练带扩展MLA机制的GRPO模型...")
    model_extended = ForecastingGRPOMLAModel(input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2, lambda_policy=lambda_policy)
    model_extended.to(device)
    optimizer_extended = torch.optim.Adam(model_extended.parameters(), lr=0.0003)
    scheduler_extended = torch.optim.lr_scheduler.StepLR(optimizer_extended, step_size=10, gamma=0.9)
    n_epochs_extended = 22
    for epoch in range(n_epochs_extended):
        train_loss_extended = train_model_full(model_extended, train_loader, optimizer_extended, device, grad_clip=1.0)
        mape_extended = validate_model(model_extended, val_loader, device, dataset_obj, debug=True)
        scheduler_extended.step()
        print(f"Epoch {epoch+1}/{n_epochs_extended} - 扩展MLA模型训练损失: {train_loss_extended:.4f}, MAPE: {mape_extended:.2f}%")
    
if __name__ == "__main__":
    main()



