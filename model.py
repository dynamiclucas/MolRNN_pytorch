# Import necessary PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Define the PyTorch version of the CVAE model
class CVAE_PyTorch(nn.Module):
    def __init__(self, vocab_size, args):
        super(CVAE_PyTorch, self).__init__()
        
        # Initialize variables
        self.vocab_size = vocab_size
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.unit_size = args['unit_size']
        self.n_rnn_layer = args['n_rnn_layer']
        
        # RNN Encoder
        self.encoder = nn.LSTM(input_size=self.vocab_size,
                               hidden_size=self.unit_size,
                               num_layers=self.n_rnn_layer,
                               batch_first=True)
        
        # Softmax layer
        self.softmax_layer = nn.Linear(self.unit_size, self.vocab_size)
        
        # Embedding layer for encoding
        self.embedding_encode = nn.Embedding(self.vocab_size, self.unit_size)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, X, L):
        # Embedding
        X_embedded = self.embedding_encode(X)
        
        # LSTM encoding
        output, _ = self.encoder(X_embedded)
        
        # Softmax layer
        decoded_logits = self.softmax_layer(output)
        
        return decoded_logits

# Sample arguments for model initialization
sample_args = {
    'batch_size': 128,
    'lr': 0.001,
    'unit_size': 512,
    'n_rnn_layer': 3
}

# Initialize the PyTorch model with sample arguments and vocab_size (assuming vocab_size=20 for demonstration)
vocab_size = 20  # This is just a placeholder for demonstration
model_pytorch = CVAE_PyTorch(vocab_size, sample_args)

# Show the model architecture
model_pytorch
