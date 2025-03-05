# Volunteer Allocation System  

## Overview  
This app assigns volunteers to tasks based on their experience and past activity. It uses AI to predict skill levels and match volunteers to the best opportunities.  

## How to Install and Run  

1. Clone the repository:  

```
git clone https://github.com/YOUR_GITHUB_USERNAME/VolunteerCloudApp.git
cd VolunteerCloudApp
```

2. Create a virtual environment:  

```
python -m venv myenv
myenv\Scripts\activate  # Windows
```

3. Install required libraries:  

```
pip install -r requirements.txt
```

4. Run the app:  

```
streamlit run app.py
```

## How It Works  
- The app predicts a volunteer’s experience level.  
- It assigns tasks based on urgency and fit.  
- Volunteers see their recommended tasks in the app.  

## Technologies  
- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
