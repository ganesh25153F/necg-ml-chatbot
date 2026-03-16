from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Your existing data (unchanged)
TEACHERS_SUBJECTS = {
    "python": "Mrs. Saritha (8 yrs exp)", "java": "Mrs. Surekha (8 yrs exp)",
    "ds": "Mr. Saabjaan (7 yrs exp)", "data structures": "Mr. Saabjaan (7 yrs exp)",
    "dbms": "Mr. Muthyaalu (15 yrs exp)", "database": "Mr. Muthyaalu (15 yrs exp)",
    "os": "Mrs. Subhashini (8 yrs exp)", "operating system": "Mrs. Subhashini (8 yrs exp)",
    "cn": "Mr. Srinivasulu (11 yrs exp)", "networks": "Mr. Srinivasulu (11 yrs exp)",
    "web": "Mr. Ramesh Reddy (10 yrs exp)", "web tech": "Mr. Ramesh Reddy (10 yrs exp)",
    "ai": "Mrs. Mobeen", "ml": "Mrs. Mobeen", "cloud": "Mr. KoteswaraRao (AWS Certified)",
    "cyber": "Mrs. Priyanjali (CEH Certified)", "circuits": "Dr. BMK (18 yrs exp)",
    "vlsi": "Mr. Kumar Swamy (12 yrs exp)", "signals": "Mrs. Dhivyavaani (14 yrs exp)",
    "embedded": "Mr. Jagadheesh (10 yrs exp)"
}

# NEW: B.Tech Courses List
BTECH_COURSES = [
    "💻 **B.Tech Computer Science & Engineering (CSE)** - 4 Years",
    "📡 **B.Tech Electronics & Communication Engineering (ECE)** - 4 Years", 
    "🔧 **B.Tech Mechanical Engineering (ME)** - 4 Years",
    "⚡ **B.Tech Electrical & Electronics Engineering (EEE)** - 4 Years",
    "🏗️ **B.Tech Civil Engineering (CE)** - 4 Years"
]

# NEW: Fee Structure
FEE_STRUCTURE = [
    "💰 **B.Tech Fee Structure (Per Year):**",
    "• **1st Year**: ₹75,000 (Tuition) + ₹10,000 (Other Fees)",
    "• **2nd Year**: ₹80,000 (Tuition) + ₹10,000 (Other Fees)", 
    "• **3rd Year**: ₹85,000 (Tuition) + ₹10,000 (Other Fees)",
    "• **4th Year**: ₹90,000 (Tuition) + ₹10,000 (Other Fees)",
    "• **Total (4 Years)**: ₹3,70,000",
    "• **Hostel**: ₹60,000/Year (Optional)",
    "• **Scholarships**: 100% for 98%+ in Intermediate"
]

CSE_FACULTY = [
    "👨‍🏫 **Dr. Venkateswara Rao** - CSE HOD (PhD IIT, AI/ML Expert)",
    "👩‍🏫 **Mrs. Saritha** - Python (8 yrs exp)", "👩‍🏫 **Mrs. Surekha** - Java (8 yrs exp)",
    "👨‍🏫 **Mr. Saabjaan** - Data Structures (7 yrs exp)", "👨‍🏫 **Mr. Muthyaalu** - DBMS (15 yrs exp)",
    "👩‍🏫 **Mrs. Subhashini** - OS (8 yrs exp)", "👩‍🏫 **Mrs. Mobeen** - AI/ML",
    "👨‍🏫 **Mr. KoteswaraRao** - Cloud Computing (AWS Certified)"
]

ECE_FACULTY = [
    "👩‍🏫 **Prof. Neeraja Reddy** - ECE HOD (PhD, Signal Processing)",
    "👨‍🏫 **Dr. BMK** - Circuits (18 yrs exp)", "👨‍🏫 **Mr. Kumar Swamy** - VLSI Design (12 yrs exp)",
    "👩‍🏫 **Mrs. Dhivyavaani** - Signals & Systems (14 yrs exp)", "👨‍🏫 **Mr. Jagadheesh** - Embedded Systems (10 yrs exp)"
]

MCA_FACULTY = [
    "👩‍🏫 **Prof. Sucharitha** - MCA HOD (13 yrs Industry Exp)",
    "👨‍🏫 **Mr. Srinivasulu** - Computer Networks (11 yrs exp)",
    "👨‍🏫 **Mr. Ramesh Reddy** - Web Tech (10 yrs exp)",
    "👩‍🏫 **Mrs. Priyanjali** - Cyber Security (CEH Certified)"
]

ALL_FACULTY_LIST = CSE_FACULTY + ECE_FACULTY + MCA_FACULTY

PORTAL_LINKS = {
    "results": "https://narayanagroup.co.in/patient/student/EngLogin.aspx",
    "attendance": "http://115.241.194.20/patient/student/student_login.aspx",
    "fees": "https://narayanagroup.co.in/onlinefeepay/razorpay/createordernew.aspx",
    "admissions": "https://ap-btech-admissions-3.onrender.com/"
}

# 🤖 ML MODEL - Scikit-learn Pipeline
class NECGMLBot:
    def __init__(self):
        self.model = self._train_model()
    
    def _train_model(self):
        training_data = {
            'greeting': ['hi', 'hello', 'hey', 'namaste', 'good morning'],
            'cse_faculty': ['cse faculty', 'cse teachers', 'computer science faculty', 'cse hod'],
            'ece_faculty': ['ece faculty', 'ece teachers', 'electronics faculty', 'ece hod'],
            'mca_faculty': ['mca faculty', 'mca teachers', 'master faculty'],
            'all_faculty': ['all faculty', 'complete faculty', 'all teachers', 'faculty list'],
            'btech_courses': ['btech courses', 'b tech courses', 'engineering courses', 'b.e courses'],  # NEW
            'fee_structure': ['fee structure', 'fees details', 'btech fees', 'course fees', 'tuition fees'],  # NEW
            'subject_search': ['python teacher', 'java faculty', 'dbms teacher', 'os professor'],
            'results': ['results', 'marks', 'grades', 'sem results'],
            'attendance': ['attendance', 'classes attended', 'attendance status'],
            'fees': ['fees', 'fee structure', 'payment', 'tuition'],
            'admissions': ['admission', 'admissions', 'apply', 'joining'],
            'courses': ['courses', 'course list', 'programs', 'btech programs']
        }
        
        X = []
        y = []
        for category, phrases in training_data.items():
            for phrase in phrases:
                X.append(phrase)
                y.append(category)
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        pipeline.fit(X, y)
        return pipeline
    
    def predict(self, message):
        try:
            prediction = self.model.predict([message.lower()])[0]
            return prediction
        except:
            return 'unknown'

# Initialize ML Bot
ml_bot = NECGMLBot()

def get_ml_response(message):
    """ML-powered response generation"""
    category = ml_bot.predict(message)
    
    # NEW: B.Tech Courses Response
    if category == 'btech_courses':
        return '🎓 **Available B.Tech Courses:**', 'show_btech_courses'
    
    # NEW: Fee Structure Response  
    elif category == 'fee_structure':
        return '💰 **B.Tech Fee Structure:**', 'show_fee_structure'
    
    # ML Decision Logic (existing)
    elif category == 'greeting':
        return 'Namaste macha! 😎 ', None
    
    elif category == 'cse_faculty':
        return '🎓 **CSE Faculty List (one by one):**', 'show_cse'
    elif category == 'ece_faculty':
        return '📡 **ECE Faculty List (one by one):**', 'show_ece'
    elif category == 'mca_faculty':
        return '💻 **MCA Faculty List (one by one):**', 'show_mca'
    elif category == 'all_faculty':
        return '👨‍🏫 **Complete NECG Faculty (one by one):**', 'show_all'
    
    elif category == 'subject_search':
        words = message.lower().split()
        for subject in TEACHERS_SUBJECTS:
            if subject in words:
                return f'📚 **{subject.upper()}**: {TEACHERS_SUBJECTS[subject]}', None
    
    elif category == 'results':
        return '📊 **Click on above results portal!**', 'results'
    elif category == 'attendance':
        return '👤 **Click on above attendance portal!**', 'attendance'
    elif category == 'fees':
        return '💳 **Click on above fee payment portal!**', 'fees'
    elif category == 'admissions':
        return '🎓 **Click on above admission portal!**', 'admissions'
    
    # Fallback pattern matching
    msg = message.lower()
    if any(word in msg for word in ['faculty', 'teacher']):
        return '👨‍🏫 **Faculty Search:**<br>• cse faculty<br>• ece faculty<br>• mca faculty<br>• all faculty', None
    if any(word in msg for word in ['btech', 'b tech', 'engineering', 'be']):
        return '🎓 **B.Tech Options:**<br>• btech courses<br>• fee structure', None
    
    return '🔍 **Quick options:** hi | btech courses | fee structure | cse faculty | results | fees | admissions', None

@app.route('/')
def home():
    return render_template('index.html', 
                         cse_faculty=CSE_FACULTY,
                         ece_faculty=ECE_FACULTY,
                         mca_faculty=MCA_FACULTY,
                         all_faculty=ALL_FACULTY_LIST,
                         portal_links=PORTAL_LINKS,
                         teachers_subjects=TEACHERS_SUBJECTS,
                         btech_courses=BTECH_COURSES,      # NEW
                         fee_structure=FEE_STRUCTURE)     # NEW

@app.route('/api/chat', methods=['POST'])
def chat():
    message = request.json.get('message', '')
    response, action = get_ml_response(message)
    return jsonify({'response': response, 'action': action})

@app.route('/requirements')
def requirements():
    return render_template('requirements.html')

if __name__ == '__main__':
    print("🚀 NECG ML CHATBOT LIVE at http://127.0.0.1:5000")
    print("✅ ML Model: scikit-learn Naive Bayes + TF-IDF Vectorizer")
    print("✅ NEW: B.Tech Courses + Fee Structure Added!")
    print("✅ Templates folder structure | All features active!")
    app.run(debug=True, port=5000)
