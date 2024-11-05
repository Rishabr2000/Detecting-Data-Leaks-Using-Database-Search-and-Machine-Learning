import os
import sys
import django
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "helloworld.settings")
django.setup()

from hello.models import UserData


data = UserData.objects.all().values('email', 'password')
df = pd.DataFrame(list(data))


df['leaked'] = 1


non_leaked_fraction = 0.3  # Adjust this fraction as needed
non_leaked_count = int(len(df) * non_leaked_fraction)
df.loc[:non_leaked_count, 'leaked'] = 0


def extract_features(df):
    df['email_length'] = df['email'].apply(len)
    df['email_domain'] = df['email'].apply(lambda x: x.split('@')[-1])
    df['email_numbers'] = df['email'].apply(lambda x: sum(c.isdigit() for c in x))
    return df

df = extract_features(df)


df = df.sample(frac=0.1, random_state=42)

# Preparing the data
X = df[['email', 'email_length', 'email_domain', 'email_numbers']]
y = df['leaked']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


numerical_features = ['email_length', 'email_numbers']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_features = ['email_domain']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


text_transformer = TfidfVectorizer()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('tfidf_email', text_transformer, 'email')
    ])


model = RandomForestClassifier(random_state=42)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])


param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5]
}


grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=3)


grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Saving trained model to a file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.pkl')
joblib.dump(best_model, model_path)


vectorizer_email_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorizer_email.pkl')
onehot_encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onehot_encoder.pkl')

joblib.dump(best_model.named_steps['preprocessor'].named_transformers_['tfidf_email'], vectorizer_email_path)
joblib.dump(best_model.named_steps['preprocessor'].named_transformers_['cat'], onehot_encoder_path)
