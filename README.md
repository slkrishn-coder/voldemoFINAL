# **README for AI-Powered Volunteer Allocation System**

## **Overview**  
This AI-powered volunteer allocation system uses machine learning models to optimize volunteer assignments based on experience level, task urgency, and past engagement. The system predicts volunteer expertise and assigns them tasks that best match their skills, preventing burnout and improving retention.

---

## **Features**
- **AI-Driven Volunteer Matching**  
  - Uses RandomForestClassifier to classify volunteers into four experience levels:  
    - **Beginner** → 0-10 tasks  
    - **Intermediate** → 11-30 tasks  
    - **Mentor** → 31-50 tasks  
    - **Site Leader** → 51+ tasks  

- **Task Fit Scoring**  
  - Uses XGBRegressor to assign volunteers a fit score (0 to 1) based on:  
    - Task urgency  
    - Past engagement  
    - Completion rates  

- **Dynamic Task Assignment**  
  - High urgency tasks → Assigned to Site Leaders & Mentors  
  - Medium urgency tasks → Assigned to Intermediate volunteers  
  - Low urgency tasks → Assigned to Beginners for skill-building  

- **Burnout Prevention & Reallocation**  
  - Volunteers nearing disengagement are reassigned to lower-stress projects  
  - Dropout risks are detected using engagement trends  

---

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/VolunteerCloudApp.git
cd VolunteerCloudApp
```

### **2. Create a Virtual Environment**
```bash
python -m venv myenv
myenv\Scripts\activate  # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Streamlit App**
```bash
streamlit run app.py
```
This will launch the application in your browser.

---

## **How It Works**
1. Data is generated to simulate volunteer engagement using NumPy.  
2. RandomForestClassifier predicts experience level.  
3. XGBRegressor calculates a fit score for task matching.  
4. Volunteers select their ID in the Streamlit UI and receive a task recommendation.  
5. Progress bars and alerts guide users on their assignments.  

---

## **Technologies Used**
- Python  
- Streamlit  
- Pandas & NumPy  
- Scikit-learn (RandomForestClassifier)  
- XGBoost (XGBRegressor)  

---

## **Future Enhancements**
- Expand AI models to incorporate volunteer feedback and task ratings  
- Deploy app online using Streamlit Cloud for easy access  
- Integrate real-world volunteer datasets for improved accuracy  
