# docker build -t loan-prediction .

# docker tag loan-prediction us-central1-docker.pkg.dev/airy-shadow-465511-v9/python-app/loan-prediction

# docker push us-central1-docker.pkg.dev/airy-shadow-465511-v9/python-app/loan-prediction

# gcloud run deploy xgboost-coupon-model --image  us-central1-docker.pkg.dev/airy-shadow-465511-v9/python-app/loan-prediction --region us-central1