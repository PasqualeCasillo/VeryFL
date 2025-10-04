import torch.nn as nn

class PowerGridNet(nn.Module):
    """
    Rete neurale robusta per classificazione binaria del Power Grid dataset.
    Architettura semplificata senza BatchNorm per evitare problemi di tipo.
    """
    
    def __init__(self, input_dim=128, class_num=2):
        super(PowerGridNet, self).__init__()
        
        # Rete semplice e stabile
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, class_num)
        )
        
        # Inizializzazione pesi
        self._init_weights()
        
    def _init_weights(self):
        """Inizializzazione robusta dei pesi"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.network(x)


def get_powergrid_net(class_num=2):
    """Factory function per ottenere il modello PowerGrid"""
    return PowerGridNet(input_dim=128, class_num=class_num)