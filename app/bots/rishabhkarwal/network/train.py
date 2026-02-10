import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import base64
import os
import sys
from tqdm import tqdm

class Config:
    # Data
    NUM_POSITIONS = 100
    STOCKFISH_DEPTH = 12
    STOCKFISH_PATH = "stockfish/stockfish.exe"
    WEIGHTS_FILE = 'weights.txt'
    
    # Architecture
    INPUT_SIZE = 768
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    
    # Training
    BATCH_SIZE = 1024           
    LEARNING_RATE = 0.002       
    NUM_EPOCHS = 50
    
    # Scaling (Sigmoid K-Factor)
    K_FACTOR = 0.004  


def get_random_board(max_moves=80):
    board = chess.Board()
    depth = np.random.randint(5, max_moves)
    for _ in range(depth):
        if board.is_game_over(): break
        legal_moves = list(board.legal_moves)
        if not legal_moves: break
        
        # 40% chance to pick a capture (makes positions tactical)
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures and np.random.rand() < 0.40:
            move = np.random.choice(captures)
        else:
            move = np.random.choice(legal_moves)
        board.push(move)
    return board

def generate_dataset(config):
    print(f"Generating {config.NUM_POSITIONS} positions")
    if not os.path.exists(config.STOCKFISH_PATH):
        print(f"ERROR: Stockfish not found at {config.STOCKFISH_PATH}")
        sys.exit(1)

    engine = chess.engine.SimpleEngine.popen_uci(config.STOCKFISH_PATH)
    inputs = []
    targets = [] # WDL probabilities (0.0 to 1.0)
    raw_cp = []  # keep raw CP for testing purposes
    
    seen_fens = set()
    
    with tqdm(total=config.NUM_POSITIONS) as pbar:
        while len(inputs) < config.NUM_POSITIONS:
            board = get_random_board()
            fen = board.fen()
            if board.is_game_over() or fen in seen_fens: continue
            
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=config.STOCKFISH_DEPTH))
                score = info["score"].relative
                
                # convert mate to a high cp value
                if score.is_mate():
                    cp = 2000 if score.mate() > 0 else -2000
                else:
                    cp = score.score()
                
                # sigmoid scaling
                wdl = 1 / (1 + np.exp(-config.K_FACTOR * cp))
                
                # encode board
                feats = np.zeros(768, dtype=np.float32)
                # white
                for pt in range(1, 7):
                    for sq in board.pieces(pt, chess.WHITE):
                        feats[(pt-1)*64 + sq] = 1.0
                # black
                for pt in range(1, 7):
                    for sq in board.pieces(pt, chess.BLACK):
                        feats[(pt+5)*64 + sq] = 1.0

                inputs.append(feats)
                targets.append(wdl)
                raw_cp.append(cp)
                seen_fens.add(fen)
                pbar.update(1)
            except: continue

    engine.quit()
    return (np.array(inputs, dtype=np.float32), 
            np.array(targets, dtype=np.float32), 
            np.array(raw_cp, dtype=np.float32))


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(768, 32)
        self.l2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x

    def predict_cp(self, x, k_factor):
        prob = self.forward(x)
        prob = torch.clamp(prob, 0.0001, 0.9999)
        return -torch.log(1/prob - 1) / k_factor

def test_accuracy(model, X, raw_cp_targets, device, config):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(device)
        
        # raw cp predictions
        pred_probs = model(X_tensor).cpu().numpy().flatten()
        
        pred_probs = np.clip(pred_probs, 0.001, 0.999)
        pred_cp = -np.log(1/pred_probs - 1) / config.K_FACTOR
        
        mask = np.abs(raw_cp_targets) > 50
        correct_sign = np.sign(pred_cp[mask]) == np.sign(raw_cp_targets[mask])
        sign_acc = np.mean(correct_sign) * 100
        
        mae = np.mean(np.abs(pred_cp - raw_cp_targets))
        
        diff = np.abs(pred_cp - raw_cp_targets)
        acc_100 = np.mean(diff < 100) * 100
        acc_300 = np.mean(diff < 300) * 100
        acc_500 = np.mean(diff < 500) * 100
        acc_800 = np.mean(diff < 800) * 100
        
        print(f"\nResults")
        print(f"Winner Prediction: {sign_acc:.2f}% (excluding draws)")
        print(f"Average CP Error: {mae:.0f} cp")
        print(f"Precision within 100cp:  {acc_100:.2f}%")
        print(f"Precision within 300cp:  {acc_300:.2f}%")
        print(f"Precision within 500cp:  {acc_500:.2f}%")
        print(f"Precision within 800cp:  {acc_800:.2f}%")
        print("-" * 30)

def train_and_export():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    X, y, raw_cp = generate_dataset(config)
    
    # split (80% Train, 20% Test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    cp_test = raw_cp[split_idx:]
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # train
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\nTraining")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Loss {total_loss/len(loader):.5f}")

    # test
    test_accuracy(model, X_test, cp_test, device, config)
    
    print("\nExporting")
    
    w1 = model.l1.weight.detach().cpu().numpy().T.flatten().astype(np.float32)
    b1 = model.l1.bias.detach().cpu().numpy().flatten().astype(np.float32)
    w2 = model.l2.weight.detach().cpu().numpy().flatten().astype(np.float32)
    b2 = model.l2.bias.detach().cpu().numpy().flatten().astype(np.float32)
    
    full_blob = np.concatenate([w1, b1, w2, b2])
    b64_str = base64.b64encode(full_blob.tobytes()).decode('utf-8')
    
    # save to file
    with open(config.WEIGHTS_FILE, "w") as f:
        f.write(b64_str)
        
    print(f"Weights saved to '{config.WEIGHTS_FILE}' ({len(b64_str)} chars).")

if __name__ == "__main__":
    train_and_export()