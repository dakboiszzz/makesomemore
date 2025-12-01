# makesomemore
Based on the makemore series of Andrej

## To Do
- Train on Vietnamese names?
- Rebuild the Adam Optimizer
- Playing around with activations functions, `block_size`, and `emb_size`, or the number of layers

## Some interesting result
- The train-val-test loss 
```
Final Evaluation:
train loss: 2.1419
val loss: 2.1641
test loss: 2.1637
```
- Train-val-test loss after adding the AdamW Optimizer
```
Final Evaluation:
train loss: 2.0827
val loss: 2.1305
test loss: 2.1274
```
- The names
```
carmah.
amelle.
khaimrix.
taty.
halayan.
jazhnel.
den.
rhy.
kaeli.
nellara.
chaiiv.
kaleigh.
ham.
porn. # Woah i mean this is wild :))
quint.
sulin.
alian.
quinterri.
jarysia.
kael.
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/dakboiszzz/makesomemore.git
cd makesomemore
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
```

### 3. Activate virtual environment
```bash
# On Mac/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python scripts/train.py
```

### Generate names
```bash
python scripts/sample.py
```

## Project Structure
```
makesomemore/
├── data/
│   └── names.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── layers.py
├── scripts/
│   ├── train.py
│   └── sample.py
├── requirements.txt
├── config.yaml
└── README.md
```