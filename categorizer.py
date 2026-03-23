import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CATEGORY_META = {
    "Food & Dining":     {"emoji": "🍔", "color": "#f97316"},
    "Transport":         {"emoji": "🚗", "color": "#3b82f6"},
    "Shopping":          {"emoji": "🛍️", "color": "#a855f7"},
    "Entertainment":     {"emoji": "🎬", "color": "#ec4899"},
    "Health & Medical":  {"emoji": "💊", "color": "#10b981"},
    "Utilities":         {"emoji": "💡", "color": "#f59e0b"},
    "Travel":            {"emoji": "✈️", "color": "#06b6d4"},
    "Education":         {"emoji": "📚", "color": "#6366f1"},
    "Groceries":         {"emoji": "🛒", "color": "#84cc16"},
    "Finance":           {"emoji": "💳", "color": "#64748b"},
    "Subscriptions":     {"emoji": "📱", "color": "#8b5cf6"},
    "Other":             {"emoji": "📦", "color": "#94a3b8"},
}

TRAINING_DATA = [
    # Food & Dining
    ("mcdonalds burger", "Food & Dining"),
    ("starbucks coffee", "Food & Dining"),
    ("pizza hut order", "Food & Dining"),
    ("uber eats delivery", "Food & Dining"),
    ("restaurant dinner", "Food & Dining"),
    ("doordash food", "Food & Dining"),
    ("subway sandwich", "Food & Dining"),
    ("sushi bar", "Food & Dining"),
    ("cafe latte", "Food & Dining"),
    ("dominos pizza", "Food & Dining"),
    ("kfc chicken", "Food & Dining"),
    ("burger king meal", "Food & Dining"),
    ("grubhub order", "Food & Dining"),
    ("lunch takeout", "Food & Dining"),
    ("dinner at restaurant", "Food & Dining"),
    ("taco bell", "Food & Dining"),
    ("chipotle burrito", "Food & Dining"),
    ("panera bread", "Food & Dining"),
    ("wendy's burger", "Food & Dining"),
    ("dunkin donuts coffee", "Food & Dining"),

    # Transport
    ("uber ride", "Transport"),
    ("lyft trip", "Transport"),
    ("gas station fill", "Transport"),
    ("shell petrol", "Transport"),
    ("bp fuel", "Transport"),
    ("metro transit pass", "Transport"),
    ("bus ticket", "Transport"),
    ("parking fee", "Transport"),
    ("car wash", "Transport"),
    ("oil change jiffy lube", "Transport"),
    ("subway metro", "Transport"),
    ("toll highway", "Transport"),
    ("taxi fare", "Transport"),
    ("bike rental", "Transport"),
    ("ola cab", "Transport"),
    ("rapido bike", "Transport"),
    ("petrol bunk", "Transport"),
    ("vehicle maintenance", "Transport"),
    ("auto rickshaw", "Transport"),
    ("train ticket", "Transport"),

    # Shopping
    ("amazon purchase", "Shopping"),
    ("walmart shopping", "Shopping"),
    ("target store", "Shopping"),
    ("zara clothes", "Shopping"),
    ("h&m clothing", "Shopping"),
    ("nike shoes", "Shopping"),
    ("adidas store", "Shopping"),
    ("flipkart order", "Shopping"),
    ("ebay item", "Shopping"),
    ("best buy electronics", "Shopping"),
    ("uniqlo shirt", "Shopping"),
    ("gap jeans", "Shopping"),
    ("old navy purchase", "Shopping"),
    ("shein order", "Shopping"),
    ("fashion nova", "Shopping"),
    ("department store", "Shopping"),
    ("mall shopping", "Shopping"),
    ("online shopping", "Shopping"),
    ("clothes purchase", "Shopping"),
    ("shoes store", "Shopping"),

    # Entertainment
    ("netflix subscription", "Entertainment"),
    ("spotify premium", "Entertainment"),
    ("movie theater ticket", "Entertainment"),
    ("amc cinema", "Entertainment"),
    ("steam game purchase", "Entertainment"),
    ("playstation store", "Entertainment"),
    ("xbox live", "Entertainment"),
    ("hulu streaming", "Entertainment"),
    ("disney plus", "Entertainment"),
    ("concert ticket", "Entertainment"),
    ("bowling alley", "Entertainment"),
    ("escape room", "Entertainment"),
    ("comedy club", "Entertainment"),
    ("live event ticket", "Entertainment"),
    ("gaming purchase", "Entertainment"),
    ("youtube premium", "Entertainment"),
    ("twitch subscription", "Entertainment"),
    ("apple tv plus", "Entertainment"),
    ("prime video", "Entertainment"),
    ("amusement park", "Entertainment"),

    # Health & Medical
    ("pharmacy cvs", "Health & Medical"),
    ("walgreens medicine", "Health & Medical"),
    ("doctor visit copay", "Health & Medical"),
    ("dental appointment", "Health & Medical"),
    ("gym membership", "Health & Medical"),
    ("planet fitness", "Health & Medical"),
    ("prescription medication", "Health & Medical"),
    ("hospital bill", "Health & Medical"),
    ("eye doctor", "Health & Medical"),
    ("vitamin supplement", "Health & Medical"),
    ("yoga class", "Health & Medical"),
    ("health insurance", "Health & Medical"),
    ("lab test fee", "Health & Medical"),
    ("physiotherapy session", "Health & Medical"),
    ("protein powder", "Health & Medical"),
    ("fitness equipment", "Health & Medical"),
    ("medical checkup", "Health & Medical"),
    ("urgent care visit", "Health & Medical"),
    ("therapy session", "Health & Medical"),
    ("chiropractor visit", "Health & Medical"),

    # Utilities
    ("electric bill payment", "Utilities"),
    ("water bill", "Utilities"),
    ("internet service provider", "Utilities"),
    ("comcast internet", "Utilities"),
    ("at&t phone bill", "Utilities"),
    ("verizon wireless", "Utilities"),
    ("gas utility bill", "Utilities"),
    ("electricity payment", "Utilities"),
    ("xfinity cable", "Utilities"),
    ("t-mobile phone", "Utilities"),
    ("spectrum internet", "Utilities"),
    ("trash collection", "Utilities"),
    ("sewer bill", "Utilities"),
    ("heating oil", "Utilities"),
    ("solar panel payment", "Utilities"),
    ("monthly phone plan", "Utilities"),
    ("broadband bill", "Utilities"),
    ("utility payment", "Utilities"),
    ("cable tv bill", "Utilities"),
    ("home internet", "Utilities"),

    # Travel
    ("flight booking delta", "Travel"),
    ("airbnb rental", "Travel"),
    ("hotel hilton stay", "Travel"),
    ("united airlines ticket", "Travel"),
    ("marriott hotel", "Travel"),
    ("expedia booking", "Travel"),
    ("booking com hotel", "Travel"),
    ("rental car enterprise", "Travel"),
    ("airport lounge", "Travel"),
    ("cruise booking", "Travel"),
    ("hostel stay", "Travel"),
    ("southwest airlines", "Travel"),
    ("international flight", "Travel"),
    ("resort booking", "Travel"),
    ("travel insurance", "Travel"),
    ("visa fee", "Travel"),
    ("passport service", "Travel"),
    ("airfare purchase", "Travel"),
    ("vacation rental", "Travel"),
    ("hertz car rental", "Travel"),

    # Education
    ("udemy course", "Education"),
    ("coursera subscription", "Education"),
    ("textbook purchase", "Education"),
    ("university tuition", "Education"),
    ("school supplies", "Education"),
    ("online class fee", "Education"),
    ("skillshare membership", "Education"),
    ("linkedin learning", "Education"),
    ("bootcamp payment", "Education"),
    ("language app duolingo", "Education"),
    ("masterclass subscription", "Education"),
    ("exam fee", "Education"),
    ("tutoring session", "Education"),
    ("programming course", "Education"),
    ("college application fee", "Education"),
    ("workshop registration", "Education"),
    ("certification exam", "Education"),
    ("khan academy donation", "Education"),
    ("study material", "Education"),
    ("notebook pens school", "Education"),

    # Groceries
    ("whole foods grocery", "Groceries"),
    ("kroger supermarket", "Groceries"),
    ("trader joes", "Groceries"),
    ("aldi store", "Groceries"),
    ("costco membership", "Groceries"),
    ("safeway grocery", "Groceries"),
    ("publix supermarket", "Groceries"),
    ("fresh produce market", "Groceries"),
    ("supermarket shopping", "Groceries"),
    ("weekly groceries", "Groceries"),
    ("milk eggs bread", "Groceries"),
    ("meat seafood purchase", "Groceries"),
    ("vegetables fruits", "Groceries"),
    ("cereal snacks", "Groceries"),
    ("baking supplies", "Groceries"),
    ("walmart groceries", "Groceries"),
    ("food lion", "Groceries"),
    ("h-e-b grocery", "Groceries"),
    ("wegmans", "Groceries"),
    ("meijer store", "Groceries"),

    # Finance
    ("bank transfer fee", "Finance"),
    ("atm withdrawal", "Finance"),
    ("credit card payment", "Finance"),
    ("loan emi payment", "Finance"),
    ("insurance premium", "Finance"),
    ("stock brokerage fee", "Finance"),
    ("paypal transaction", "Finance"),
    ("venmo transfer", "Finance"),
    ("western union", "Finance"),
    ("wire transfer", "Finance"),
    ("investment deposit", "Finance"),
    ("retirement contribution", "Finance"),
    ("tax payment irs", "Finance"),
    ("mortgage payment", "Finance"),
    ("rent payment", "Finance"),
    ("savings deposit", "Finance"),
    ("mutual fund", "Finance"),
    ("crypto exchange", "Finance"),
    ("overdraft fee", "Finance"),
    ("foreign exchange", "Finance"),

    # Subscriptions
    ("apple icloud", "Subscriptions"),
    ("google one storage", "Subscriptions"),
    ("microsoft 365", "Subscriptions"),
    ("dropbox plus", "Subscriptions"),
    ("adobe creative cloud", "Subscriptions"),
    ("amazon prime", "Subscriptions"),
    ("password manager 1password", "Subscriptions"),
    ("antivirus renewal", "Subscriptions"),
    ("vpn service", "Subscriptions"),
    ("notion subscription", "Subscriptions"),
    ("figma plan", "Subscriptions"),
    ("grammarly premium", "Subscriptions"),
    ("canva pro", "Subscriptions"),
    ("slack workspace", "Subscriptions"),
    ("zoom meeting plan", "Subscriptions"),
    ("github pro", "Subscriptions"),
    ("cloud hosting", "Subscriptions"),
    ("domain renewal", "Subscriptions"),
    ("saas tool subscription", "Subscriptions"),
    ("software license", "Subscriptions"),
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')

class ExpenseCategorizer:
    def __init__(self):
        self.model = None
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.train()

    def train(self):
        texts = [t[0] for t in TRAINING_DATA]
        labels = [t[1] for t in TRAINING_DATA]

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                analyzer='word'
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver='lbfgs'
            ))
        ])
        self.model.fit(texts, labels)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, description: str) -> dict:
        desc_lower = description.lower()
        proba = self.model.predict_proba([desc_lower])[0]
        classes = self.model.classes_
        top_idx = np.argmax(proba)
        category = classes[top_idx]
        confidence = round(float(proba[top_idx]) * 100, 1)
        meta = CATEGORY_META.get(category, {"emoji": "📦", "color": "#94a3b8"})
        return {
            'category': category,
            'confidence': confidence,
            'emoji': meta['emoji'],
            'color': meta['color']
        }

    def get_all_categories(self):
        return list(CATEGORY_META.keys())
