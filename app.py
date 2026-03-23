from flask import Flask, render_template, request, jsonify
from categorizer import ExpenseCategorizer
import json

app = Flask(__name__)
categorizer = ExpenseCategorizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.get_json()
    description = data.get('description', '').strip()
    amount = data.get('amount', 0)

    if not description:
        return jsonify({'error': 'Description is required'}), 400

    result = categorizer.predict(description)
    return jsonify({
        'description': description,
        'amount': amount,
        'category': result['category'],
        'confidence': result['confidence'],
        'emoji': result['emoji']
    })

@app.route('/categorize_bulk', methods=['POST'])
def categorize_bulk():
    data = request.get_json()
    transactions = data.get('transactions', [])
    results = []
    for txn in transactions:
        desc = txn.get('description', '').strip()
        amount = txn.get('amount', 0)
        if desc:
            result = categorizer.predict(desc)
            results.append({
                'description': desc,
                'amount': amount,
                'category': result['category'],
                'confidence': result['confidence'],
                'emoji': result['emoji']
            })
    return jsonify({'results': results})

@app.route('/train', methods=['POST'])
def train():
    categorizer.train()
    return jsonify({'message': 'Model trained successfully!'})

@app.route('/stats', methods=['POST'])
def stats():
    data = request.get_json()
    transactions = data.get('transactions', [])
    category_totals = {}
    for txn in transactions:
        cat = txn.get('category', 'Other')
        amt = float(txn.get('amount', 0))
        category_totals[cat] = category_totals.get(cat, 0) + amt
    return jsonify({'stats': category_totals})

if __name__ == '__main__':
    app.run(debug=True)
