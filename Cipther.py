import json
import random
import string

def caesar_cipher(text, shift):
    result = ""
    for i in range(len(text)):
        char = text[i]
        if char.isupper():
            result += chr((ord(char) + shift - 65) % 26 + 65)
        else:
            result += chr((ord(char) + shift - 97) % 26 + 97)
    return result

def vigenere_cipher(text, key):
    key = key.lower()
    result = []
    key_index = 0
    for char in text:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - 97
            if char.islower():
                result.append(chr((ord(char) + shift - 97) % 26 + 97))
            else:
                result.append(chr((ord(char) + shift - 65) % 26 + 65))
            key_index += 1
        else:
            result.append(char)
    return ''.join(result)

def substitution_cipher(text):
    alphabet = string.ascii_lowercase
    shuffled_alphabet = ''.join(random.sample(alphabet, len(alphabet)))
    table = str.maketrans(alphabet, shuffled_alphabet)
    return text.translate(table)

def transposition_cipher(text, key):
    n = len(text)
    key_index = list(range(key))
    table = [''] * key
    for i in range(n):
        table[i % key] += text[i]
    return ''.join(table)

def generate_dataset():
    data = []
    plaintexts = [
        "hello world", "data science", "mixture of experts", "cipher breaking", "neural networks",
        "machine learning", "artificial intelligence", "deep learning", "natural language processing",
        "computer vision", "data analysis", "predictive modeling", "data mining", "statistical learning",
        "reinforcement learning", "supervised learning", "unsupervised learning", "feature engineering",
        "model evaluation", "algorithm development", "training data", "test data", "cross-validation",
        "hyperparameter tuning", "gradient descent", "neural architecture", "convolutional neural networks",
        "recurrent neural networks", "transformer models", "attention mechanisms", "data preprocessing",
        "feature extraction", "data cleaning", "text classification", "sentiment analysis", "image recognition",
        "speech recognition", "time series forecasting", "anomaly detection", "clustering algorithms",
        "dimensionality reduction", "principal component analysis", "support vector machines", "decision trees",
        "ensemble methods", "random forests", "gradient boosting", "k-means clustering", "naive bayes",
        "logistic regression", "linear regression", "bayesian inference", "deep reinforcement learning",
        "model interpretability", "fairness in AI", "bias in machine learning", "ethical AI", "AI ethics",
        "robotics", "autonomous systems", "smart devices", "Internet of Things", "edge computing",
        "cloud computing", "big data", "data lakes", "data warehouses", "data governance", "privacy protection",
        "cybersecurity", "blockchain technology", "cryptography", "secure data transmission", "digital forensics",
        "quantum computing", "high-performance computing", "supercomputers", "parallel computing", "distributed systems",
        "database management", "SQL databases", "NoSQL databases", "graph databases", "data integration",
        "data warehousing", "ETL processes", "data visualization", "dashboard design", "interactive plots",
        "business intelligence", "market analysis", "financial modeling", "risk assessment", "portfolio management",
        "economic forecasting", "healthcare analytics", "biomedical data", "genomic data", "pharmaceutical research",
        "clinical trials", "medical imaging", "patient monitoring", "disease prediction", "public health",
        "social network analysis", "online behavior", "user profiling", "recommendation systems", "personalized marketing",
        "customer segmentation", "A/B testing", "user experience research", "product analytics", "e-commerce analytics",
        "sales forecasting", "demand prediction", "inventory management", "supply chain optimization", "logistics",
        "transportation systems", "smart cities", "urban planning", "environmental monitoring", "sustainability",
        "climate change", "energy management", "renewable energy", "solar power", "wind energy", "electric vehicles",
        "automated systems", "IoT devices", "sensor networks", "data collection", "data storage", "data access",
        "data security", "access control", "data encryption", "information retrieval", "search engines",
        "web scraping", "data extraction", "content analysis", "text mining", "data science projects", "AI research",
        "technology trends", "innovation", "digital transformation", "startup culture", "entrepreneurship",
        "business strategy", "project management", "agile methodologies", "scrum", "kanban", "software development",
        "programming languages", "Python programming", "R programming", "JavaScript", "Java", "C++", "C#", "Swift",
        "Go programming", "functional programming", "object-oriented programming", "web development", "mobile apps",
        "desktop applications", "game development", "UI/UX design", "human-computer interaction", "user interfaces",
        "software engineering", "quality assurance", "testing frameworks", "continuous integration", "continuous deployment",
        "version control", "Git", "GitHub", "Bitbucket", "collaboration tools", "team communication", "remote work",
        "virtual teams", "digital collaboration", "online tools", "productivity", "time management", "work-life balance"
    ]
    # plaintexts = ["hello world", "data science", "mixture of experts", "cipher breaking", "neural networks"]
    for text in plaintexts:
        # Caesar Cipher
        shift = random.randint(1, 25)
        ciphertext = caesar_cipher(text, shift)
        data.append({"ciphertext": ciphertext, "plaintext": text, "cipher_type": "caesar"})
        
        # Vigenère Cipher
        key = ''.join(random.choices(string.ascii_lowercase, k=5))
        ciphertext = vigenere_cipher(text, key)
        data.append({"ciphertext": ciphertext, "plaintext": text, "cipher_type": "vigenere"})
        
        # Substitution Cipher
        ciphertext = substitution_cipher(text)
        data.append({"ciphertext": ciphertext, "plaintext": text, "cipher_type": "substitution"})
        
        # Transposition Cipher
        key = random.randint(2, 10)
        ciphertext = transposition_cipher(text, key)
        data.append({"ciphertext": ciphertext, "plaintext": text, "cipher_type": "transposition"})
        
        # Add more ciphers if needed...
    
    with open('cipher_dataset.json', 'w') as f:
        json.dump(data, f, indent=4)

generate_dataset()
