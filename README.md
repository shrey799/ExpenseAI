# 💰 Expense Categorizer — AI/ML Project

An ML-powered web app that automatically categorizes your transactions using NLP (TF-IDF + Logistic Regression).

## 🧠 How It Works
- Extracts text features from transaction descriptions using **TF-IDF** (Term Frequency-Inverse Document Frequency)
- Classifies into 12 categories using **Logistic Regression** (multinomial)
- Model is trained on 240+ labeled transactions and auto-saved as `models/model.pkl`

## 📂 Project Structure
```
expense-categorizer/
├── app.py              # Flask web server + API routes
├── categorizer.py      # ML model (TF-IDF + LogReg pipeline)
├── requirements.txt    # Python dependencies
├── models/             # Saved model (auto-created)
│   └── model.pkl
└── templates/
    └── index.html      # Full frontend UI
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

## ✨ Features
- **Single categorization** — type a description, get instant category + confidence
- **Bulk import** — paste multiple transactions (description, amount format)
- **Dashboard** — spending breakdown with animated bar chart
- **12 categories**: Food & Dining, Transport, Shopping, Entertainment, Health & Medical, Utilities, Travel, Education, Groceries, Finance, Subscriptions, Other

## 🛠️ API Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/categorize` | Categorize single transaction |
| POST | `/categorize_bulk` | Categorize multiple transactions |
| POST | `/stats` | Get totals by category |
| POST | `/train` | Retrain the model |

Screenshots<img width="869" height="934" alt="image" src="https://github.com/user-attachments/assets/69cc161d-f74e-4d99-8e1a-f7b1cb09264c" />


## 🔧 Extending the Project
- Add a CSV upload feature for bank statement imports
- Connect to a real database (SQLite/PostgreSQL) to persist transactions
- Add more training data to `TRAINING_DATA` in `categorizer.py`
- Try replacing LogReg with a fine-tuned BERT model for higher accuracy
- Add user authentication with Flask-Login
