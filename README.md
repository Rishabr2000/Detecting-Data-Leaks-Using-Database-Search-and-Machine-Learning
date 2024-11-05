# Detecting-Data-Leaks-Using-Database-Search-and-Machine-Learning

This project focuses on developing a robust system to detect data leaks using a two-step approach that combines traditional database search techniques with machine learning models. By leveraging a comprehensive email dataset, the system applies database search methods to identify known breaches and employs a Random Forest classifier to predict potential leaks when no prior breach is recorded. The model achieved an accuracy of 74.97%, showing its effectiveness in real-world applications for data leak detection.

Key components include:

Database Search: Quickly identifies known breaches based on predefined patterns.
Machine Learning Integration: Utilizes Random Forest classifiers, optimized with GridSearchCV for hyperparameter tuning, to predict data leak likelihood for unknown cases.
Real-Time Detection: Designed for efficient operation using Django, Pandas, and Scikit-learn to support real-time threat analysis and alerting.
This system provides a scalable and efficient solution to enhance data security, addressing both current and future cyber threats.
