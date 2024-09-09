import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import string
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

vocab = list(string.ascii_lowercase + ' ')
vocab_size = len(vocab)

char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

def encode_text(text):
    return [char_to_idx[char] for char in text if char in char_to_idx]

def decode_text(indices):
    return ''.join([idx_to_char[idx] for idx in indices])

with open('cipher_dataset.json', 'r') as f:
    data = json.load(f)

encoded_data = [
    (encode_text(item['ciphertext']), encode_text(item['plaintext']), item['cipher_type'])
    for item in data
]

train_data, temp_data = train_test_split(encoded_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

class CipherDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ciphertext, plaintext, cipher_type = self.data[idx]
        return torch.tensor(ciphertext, dtype=torch.long), torch.tensor(plaintext, dtype=torch.long), cipher_type
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    ciphertexts, plaintexts, additional_info = zip(*batch)
    
    max_length = max(max(len(c) for c in ciphertexts), max(len(p) for p in plaintexts))
    
    ciphertexts_padded = [torch.nn.functional.pad(c.clone().detach(), (0, max_length - len(c)), value=0) for c in ciphertexts]
    plaintexts_padded = [torch.nn.functional.pad(p.clone().detach(), (0, max_length - len(p)), value=0) for p in plaintexts]
    
    ciphertexts_padded = torch.stack(ciphertexts_padded)
    plaintexts_padded = torch.stack(plaintexts_padded)
    
    return ciphertexts_padded, plaintexts_padded, additional_info
train_dataset = CipherDataset(train_data)
val_dataset = CipherDataset(val_data)
test_dataset = CipherDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

def check_invalid_indices(data, vocab_size):
    for i, (ciphertext, plaintext, _) in enumerate(data):
        ciphertext_tensor = torch.tensor(ciphertext)
        plaintext_tensor = torch.tensor(plaintext)
        if ciphertext_tensor.numel() > 0:
            if torch.max(ciphertext_tensor) >= vocab_size or torch.min(ciphertext_tensor) < 0:
                print(f"Invalid index in ciphertext at sample {i}: {ciphertext_tensor}")
        else:
            print(f"Empty ciphertext tensor at sample {i}")

        if plaintext_tensor.numel() > 0:
            if torch.max(plaintext_tensor) >= vocab_size or torch.min(plaintext_tensor) < 0:
                print(f"Invalid index in plaintext at sample {i}: {plaintext_tensor}")
        else:
            print(f"Empty plaintext tensor at sample {i}")


check_invalid_indices(train_data, vocab_size)
check_invalid_indices(val_data, vocab_size)
check_invalid_indices(test_data, vocab_size)

def filter_empty_samples(dataset):
    return CipherDataset([(c, p, t) for c, p, t in dataset.data if len(c) > 0 and len(p) > 0])

train_dataset = filter_empty_samples(train_dataset)
val_dataset = filter_empty_samples(val_dataset)
test_dataset = filter_empty_samples(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("Preprocessing complete.")

class GatingNetwork(nn.Module):
    def __init__(self, embedding_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_experts)
    
    def forward(self, x):
        return torch.softmax(self.fc(x.float()), dim=-1)
    
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Expert(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Expert, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        x = self.embedding(x)
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x_packed, _ = self.lstm(x_packed)
        x, _ = pad_packed_sequence(x_packed, batch_first=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MoE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_experts):
        super(MoE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, vocab_size) for _ in range(num_experts)])
        self.gating = nn.Linear(hidden_size, num_experts)
    def forward(self, x, lengths=None):
        x = self.embedding(x)
        if lengths is not None:
            x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(x_packed)
            x, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            x, _ = self.lstm(x)
        expert_outputs = [expert(x) for expert in self.experts]
        gating_weights = torch.softmax(self.gating(x[:, -1, :]), dim=-1)
        combined_output = sum(w.unsqueeze(1).unsqueeze(2) * out for w, out in zip(gating_weights.transpose(0, 1), expert_outputs))
        return combined_output, expert_outputs, gating_weights
    
vocab_size = len(vocab)
embedding_dim = 64
hidden_dim = 128
num_experts = 10

model_path = 'Cipher_Exp10.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MoE(vocab_size, embedding_dim, hidden_dim, num_experts).to(device)
model.load_state_dict(torch.load(model_path))
print("Model Mounted :",model_path)

device = torch.device('cuda')
num_epochs = 1000
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        for ciphertext, plaintext, _ in train_loader:
            lengths = torch.sum(ciphertext != 0, dim=1)
            if torch.any(lengths == 0):
                continue
            ciphertext, plaintext = ciphertext.to(device), plaintext.to(device)
            optimizer.zero_grad()
            combined_output, expert_outputs, gating_weights = model(ciphertext, lengths)
            batch_size, seq_len, vocab_size = combined_output.size()
            combined_output = combined_output.contiguous().view(-1, vocab_size)
            plaintext = plaintext.contiguous().view(-1)
            
            min_length = min(combined_output.size(0), plaintext.size(0))
            combined_output = combined_output[:min_length]
            plaintext = plaintext[:min_length]
            
            mask = plaintext != 0
            combined_output = combined_output[mask]
            plaintext = plaintext[mask]
        
            loss = criterion(combined_output, plaintext)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ciphertext, plaintext, _ in val_loader:
                lengths = torch.sum(ciphertext != 0, dim=1)
                ciphertext, plaintext = ciphertext.to(device), plaintext.to(device)
                
                combined_output, expert_outputs, gating_weights = model(ciphertext, lengths)
                
                batch_size, seq_len, vocab_size = combined_output.size()
                combined_output = combined_output.contiguous().view(-1, vocab_size)
                plaintext = plaintext.contiguous().view(-1)
                
                min_length = min(combined_output.size(0), plaintext.size(0))
                combined_output = combined_output[:min_length]
                plaintext = plaintext[:min_length]

                mask = plaintext != 0
                combined_output = combined_output[mask]
                plaintext = plaintext[mask]
                loss = criterion(combined_output, plaintext)
                val_loss += loss.item()
        avg_loss = val_loss / len(val_loader)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}")


train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'Cipher_Exp10.pth')
print(f"Model_Saved {model_path}")

def process_ciphertext(model, ciphertext, device):
    model.eval()
    with torch.no_grad():
        encoded_ciphertext = encode_text(ciphertext)
        ciphertext_tensor = torch.tensor(encoded_ciphertext).unsqueeze(0).to(device)
        lengths = torch.tensor([len(encoded_ciphertext)])
        
        combined_output, expert_outputs, gating_weights = model(ciphertext_tensor, lengths)

        combined_plaintext = decode_text(combined_output.argmax(dim=-1).squeeze().tolist())
        expert_plaintexts = [decode_text(output.argmax(dim=-1).squeeze().tolist()) for output in expert_outputs]
        return combined_plaintext, expert_plaintexts, gating_weights.squeeze().tolist()

def test_cipher_types(model, device):
    cipher_types = ["caesar", "vigenere", "substitution", "transposition"]
    
    for cipher_type in cipher_types:
        for ciphertext, plaintext, sample_cipher_type in test_dataset:
            if sample_cipher_type == cipher_type:
                ciphertext = decode_text(ciphertext.tolist())
                plaintext = decode_text(plaintext.tolist())
                
                combined_output, expert_outputs, gating_weights = process_ciphertext(model, ciphertext, device)
                
                print(f"\nCipher Type: {cipher_type}")
                print(f"Original Ciphertext: {ciphertext}")
                print(f"Actual Plaintext: {plaintext}")
                print(f"Combined Model Output: {combined_output}")
                print("Expert Outputs:")
                for i, output in enumerate(expert_outputs):
                    print(f"  Expert {i+1}: {output}")
                print("Gating Weights:", gating_weights)
                
                break

model.eval()
test_cipher_types(model, device)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ciphertext, plaintext, _ in test_loader:
            ciphertext, plaintext = ciphertext.to(device), plaintext.to(device)
  
            combined_output, _, _ = model(ciphertext)
            probabilities = torch.softmax(combined_output, dim=-1)
            
            _, predicted = torch.max(probabilities, -1)
            
            total += plaintext.numel()
            correct += (predicted == plaintext).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy*100:.4f}')
from torchviz import make_dot
evaluate(model, test_loader)

def visualize_model(model, example_input):
    model.eval()
    with torch.no_grad():
        combined_output, expert_outputs, gating_weights = model(example_input)
        
        dot = make_dot(combined_output, params=dict(model.named_parameters()))
        dot.render('model_architecture', format='png')
        dot.view()

# example_input = torch.tensor([encode_text('example')]).to(device)
# visualize_model(model, example_input)
