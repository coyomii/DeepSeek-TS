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

## load data
df = pd.read_csv("sales_data.csv")

df.plot(x="date", y=["product1_sales", "product2_sales", "product3_sales", "product4_sales", "product5_sales"], figsize=(12, 6))
plt.title("Simulated Sales Data for 5 Products")
plt.show()

# -------------------------------
#  Data Preparation for Time Series Forecasting with Normalization
# -------------------------------
class SalesDataset(Dataset):
    def __init__(self, df, input_window=30, forecast_horizon=5):
        """
        df: DataFrame with columns: date, product1_sales, ..., product5_sales
        input_window: number of days used as input
        forecast_horizon: number of days to forecast; target = avg sales over these days per product
        """
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        self.df = df.reset_index(drop=True)
        # Use only the sales columns (all except 'date')
        data = df.drop(columns=['date']).values.astype(np.float32)
        
        # Compute normalization parameters on the entire dataset
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-6  # avoid division by zero
        
        # Normalize the data
        self.data = (data - self.mean) / self.std
        self.n_samples = len(self.data) - input_window - forecast_horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Input: sales for input_window days (normalized)
        x = self.data[idx: idx + self.input_window]
        # Target: average sales over the next forecast_horizon days (normalized)
        y = np.mean(self.data[idx + self.input_window: idx + self.input_window + self.forecast_horizon], axis=0)
        return x, y

def prepare_dataloaders(df, input_window=30, forecast_horizon=5, batch_size=32, train_ratio=0.8):
    dataset = SalesDataset(df, input_window, forecast_horizon)
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    # Chronological (out-of-time) split: first n_train samples for training, remaining for validation.
    train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(n_train, n_total)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = prepare_dataloaders(df, input_window=30, forecast_horizon=5, batch_size=32, train_ratio=0.8)

# -------------------------------
#  Model Components: Simple GRU-based Forecasting Model with GRPO-inspired Framework
# -------------------------------
class ForecastingGRPOModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2, lambda_policy=0.1):
        """
        This model forecasts the average sales of the next 'forecast_horizon' days for each product
        using a GRU encoder. It includes a policy branch to compute a GRPO-inspired loss.
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
# Training and Validation Functions (Using MAPE as Error Metric)
# -------------------------------
def train_model_full(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)  # (batch, seq_len, input_dim)
        y = y.to(device)  # (batch, num_products)
        optimizer.zero_grad()
        forecast, policy_value = model(x)
        # Forecast loss: use a combination of MSE and MAE
        loss_mse = F.mse_loss(forecast, y)
        loss_mae = F.l1_loss(forecast, y)
        loss_forecast = 0.5 * loss_mse + 0.5 * loss_mae
        
        # GRPO-inspired policy loss:
        # Compute advantage as the mean error over products for each sample.
        advantage = (y - forecast).mean(dim=1, keepdim=True)
        baseline = 0.5  # chosen constant baseline
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

def validate_model(model, dataloader, device, dataset_obj, debug=False):
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
    # Invert normalization: forecast_orig = forecast * std + mean
    mean = dataset_obj.mean
    std = dataset_obj.std
    all_preds_orig = all_preds * std + mean
    all_targets_orig = all_targets * std + mean
    if debug:
        print("Prediction range:", np.min(all_preds_orig), np.max(all_preds_orig))
        print("Target range:", np.min(all_targets_orig), np.max(all_targets_orig))
    mape = np.mean(np.abs((all_targets_orig - all_preds_orig) / (all_targets_orig + 1e-6))) * 100
    return mape

# -------------------------------
# ARMA Forecasting for Comparison
# -------------------------------
def arma_forecast(series, forecast_horizon):
    """
    Fits an ARIMA(1,0,1) model on the provided series and forecasts forecast_horizon steps ahead.
    Returns the average forecast.
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
    For each product (column), use a rolling ARMA forecast over the validation period on the raw data.
    Returns a dictionary of MAPE values per product and the overall average MAPE.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    products = [col for col in df.columns if col != "date"]
    mape_dict = {}
    all_mapes = []
    
    # Rolling forecast: start from i = (train_end - input_window) to (n - input_window - forecast_horizon + 1)
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
# Main Training
# -------------------------------
def main():
    # Use generated sales data
    df = generate_realistic_sales_data(n_rows=600, seed=42)
    
    # Prepare out-of-time (chronological) dataloaders (first 80% for training, remaining for validation)
    train_loader, val_loader = prepare_dataloaders(df, input_window=30, forecast_horizon=5, batch_size=32, train_ratio=0.8)
    
    # For validation inversion, we need access to the dataset normalization parameters
    dataset_obj = SalesDataset(df, input_window=30, forecast_horizon=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model parameters
    input_dim = train_loader.dataset[0][0].shape[-1]  # e.g., 5 products
    hidden_dim = 256      # Hidden dimension for GRU
    num_products = input_dim  # Predict average sales for each product
    forecast_horizon = 10  # Note: This parameter is used in the models, even though targets are for next 5 days
    lambda_policy = 0.06   # Weight for GRPO-inspired policy loss
    
    # -------------------------------
    # GRPO-inspired Forecasting Model (Existing)
    # -------------------------------
    model = ForecastingGRPOModel(input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2, lambda_policy=lambda_policy)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    n_epochs = 22

    print("Training GRPO-inspired Forecasting Model...")
    for epoch in range(n_epochs):
        train_loss = train_model_full(model, train_loader, optimizer, device, grad_clip=1.0)
        mape = validate_model(model, val_loader, device, dataset_obj, debug=True)
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs} - GRPO Model Train Loss: {train_loss:.4f}, MAPE: {mape:.2f}%")
    
    # -------------------------------
    # ARMA Evaluation for Comparison
    # -------------------------------
    print("\nEvaluating ARMA Forecasting on raw data...")
    arma_mapes, overall_arma_mape = evaluate_arma(df, input_window=30, forecast_horizon=5, train_ratio=0.8)
    print("ARMA MAPE per product:", arma_mapes)
    print("Overall ARMA MAPE:", overall_arma_mape, "%")
    
    # -------------------------------
    # Simple GRU Forecasting Model for Comparison
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

    def train_simple_model(model, dataloader, optimizer, device, grad_clip=1.0):
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

    print("\nTraining Simple GRU Forecasting Model for Comparison...")
    simple_model = SimpleGRUForecastingModel(input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2)
    simple_model.to(device)
    optimizer_simple = torch.optim.Adam(simple_model.parameters(), lr=0.0003)
    scheduler_simple = torch.optim.lr_scheduler.StepLR(optimizer_simple, step_size=10, gamma=0.9)
    n_epochs_simple = 22
    for epoch in range(n_epochs_simple):
        train_loss_simple = train_simple_model(simple_model, train_loader, optimizer_simple, device, grad_clip=1.0)
        simple_mape = validate_simple_model(simple_model, val_loader, device, dataset_obj, debug=True)
        scheduler_simple.step()
        print(f"Epoch {epoch+1}/{n_epochs_simple} - Simple GRU Train Loss: {train_loss_simple:.4f}, MAPE: {simple_mape:.2f}%")
    
    # -------------------------------
    # GRPO-inspired Forecasting with Extended MLA (Mamba-style) Mechanism
    # -------------------------------
    class ForecastingGRPOMLAModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.3, lambda_policy=0.06):
            """
            This model extends the GRPO-inspired forecasting approach by incorporating an
            extended MLA (Mamba-style) mechanism. The latent state is updated in a state-space
            manner with a nonlinear activation applied to the entire update.
            """
            super(ForecastingGRPOMLAModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.lambda_policy = lambda_policy
            self.dropout = nn.Dropout(dropout)
            # Map the input to the latent space.
            self.input_transform = nn.Linear(input_dim, hidden_dim)
            # State-space transition matrix (M)
            self.M = nn.Linear(hidden_dim, hidden_dim, bias=False)
            # Nonlinear activation function for the complete state update.
            self.activation = nn.ReLU()
            self.fc_forecast = nn.Linear(hidden_dim, num_products)
            self.policy_net = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            batch_size, seq_len, _ = x.size()
            # Initialize latent state as zeros.
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            # Iteratively update latent state with a state-space update.
            for t in range(seq_len):
                x_t = x[:, t, :]  # (batch, input_dim)
                # Compute the correction without activation first.
                correction = self.input_transform(x_t)
                # Update latent state using the ReLU activation applied to the entire sum.
                h = self.activation(self.M(h) + correction)
                h = self.dropout(h)
            forecast = self.fc_forecast(h)
            policy_value = self.policy_net(h)
            return forecast, policy_value

    print("\nTraining GRPO-inspired Forecasting Model with Extended MLA (Mamba-style) Mechanism...")
    model_extended = ForecastingGRPOMLAModel(input_dim, hidden_dim, num_products, forecast_horizon, dropout=0.2, lambda_policy=lambda_policy)
    model_extended.to(device)
    optimizer_extended = torch.optim.Adam(model_extended.parameters(), lr=0.0003)
    scheduler_extended = torch.optim.lr_scheduler.StepLR(optimizer_extended, step_size=10, gamma=0.9)
    n_epochs_extended = 22
    for epoch in range(n_epochs_extended):
        train_loss_extended = train_model_full(model_extended, train_loader, optimizer_extended, device, grad_clip=1.0)
        mape_extended = validate_model(model_extended, val_loader, device, dataset_obj, debug=True)
        scheduler_extended.step()
        print(f"Epoch {epoch+1}/{n_epochs_extended} - Extended MLA Model Train Loss: {train_loss_extended:.4f}, MAPE: {mape_extended:.2f}%")
    
if __name__ == "__main__":
    main()


