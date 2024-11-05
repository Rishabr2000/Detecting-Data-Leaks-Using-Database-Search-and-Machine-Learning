import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
import joblib
import pandas as pd
from django.conf import settings
from scipy.sparse import csr_matrix, hstack
from .models import UserData

# Loading the pre-trained model and vectorizers
model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'trained_model.pkl')
vectorizer_email_path = os.path.join(settings.BASE_DIR, 'ml_model', 'vectorizer_email.pkl')
onehot_encoder_path = os.path.join(settings.BASE_DIR, 'ml_model', 'onehot_encoder.pkl')

model = joblib.load(model_path)
tfidf_email = joblib.load(vectorizer_email_path)
onehot_encoder = joblib.load(onehot_encoder_path)

def hello_world(request):
    return HttpResponse("Hello, World!")

def new_view(request):
    return HttpResponse("This is the new link")

def check_leak_view(request):
    db_result = None
    ml_prediction = None
    ml_prediction_proba = None
    email = None

    if request.method == 'POST':
        email = request.POST.get('email')

        
        if not email:
            return HttpResponse("Email field cannot be empty.", status=400)

        # Checking in database
        try:
            db_result = UserData.objects.get(email=email)
        except UserData.DoesNotExist:
            db_result = None

        
        
        input_df = pd.DataFrame({'email': [email]})
        input_df['email_length'] = input_df['email'].apply(len)
        input_df['email_domain'] = input_df['email'].apply(lambda x: x.split('@')[-1])
        input_df['email_numbers'] = input_df['email'].apply(lambda x: sum(c.isdigit() for c in x))

        
        X_tfidf_email = tfidf_email.transform(input_df['email'])
        X_email_domain = onehot_encoder.transform(input_df[['email_domain']])

        
        X_other = csr_matrix(input_df[['email_length', 'email_numbers']].values)

        
        X_combined = hstack((X_other, X_email_domain, X_tfidf_email))

        
        ml_prediction = model.named_steps['classifier'].predict(X_combined)[0]
        ml_prediction_proba = round(model.named_steps['classifier'].predict_proba(X_combined)[0][1] * 100, 2)  # Get the probability of being leaked

    return render(request, 'hello/index.html', {'db_result': db_result, 'ml_prediction': ml_prediction, 'ml_prediction_proba': ml_prediction_proba, 'email': email})

def redirect_to_check_leak(request):
    return redirect('check_leak')  # Redirecting to the 'check_leak' view
