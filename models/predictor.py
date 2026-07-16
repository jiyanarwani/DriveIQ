import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger("driveiq.models.predictor")

class CNNLSTM_Predictor(nn.Module):
    def __init__(self, num_features=8, hidden_dim=32, num_layers=1):
        super(CNNLSTM_Predictor, self).__init__()
        
        # 1D CNN: maps (batch, num_features, seq_len) -> (batch, hidden_dim, seq_len)
        self.conv1d = nn.Conv1d(
            in_channels=num_features, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()
        
        # LSTM: maps (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # FC Layer: maps hidden_dim -> single prediction float
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x expected shape: (batch, seq_len, num_features)
        
        # PyTorch Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)
        
        # Apply Cov1D
        x = self.conv1d(x)
        x = self.relu(x)
        
        # Transpose back for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Only take the very last sequence output for the future prediction natively
        last_out = lstm_out[:, -1, :]
        
        pred = self.fc(last_out)
        return pred

def train_dummy_model():
    """Generates an initial untrained/random weights model checkpoint."""
    OUT = Path(__file__).resolve().parent
    OUT.mkdir(parents=True, exist_ok=True)
    
    model = CNNLSTM_Predictor(num_features=8)
    # Put model in eval mode specifically since it's used for inference immediately
    model.eval()
    
    # Save purely the state_dict safely
    model_path = OUT / "lstm_predictor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[predictor] Saved dummy initialized weights -> {model_path}")

if __name__ == "__main__":
    train_dummy_model()
