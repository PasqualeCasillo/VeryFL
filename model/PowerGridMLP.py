# model/PowerGridMLP.py
import torch
import torch.nn as nn

class PowerGridMLP(nn.Module):
    """
    MLP per classificazione binaria PowerGrid.
    
    Architecture:
        Input (128) -> FC(256) -> ReLU -> Dropout(0.3) ->
        FC(128) -> ReLU -> Dropout(0.3) ->
        FC(64) -> ReLU -> Dropout(0.2) ->
        FC(2) -> Output
    """
    
    def __init__(self, input_dim=128, class_num=2):
        super(PowerGridMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(64, class_num)
        )
        
        # Inizializzazione pesi
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Flatten se necessario
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


def get_powergrid_mlp(class_num=2, input_dim=128):
    """Factory function per PowerGrid MLP"""
    return PowerGridMLP(input_dim=input_dim, class_num=class_num)