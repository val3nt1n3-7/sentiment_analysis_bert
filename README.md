Sentiment analysis using BERT model.
The model is set to be trained only 1 epoch.
To improve accuracy of the model change the "MAX_LEN" in config.py and "nrows" in train.py, line 19

1. build docker container
docker build -f Dockerfile -t docker_tutorial .

2. mount folder with data for training, activate enviroment and train the model. Model will be saved in docker_data
docker run -v /home/user/Desktop/sentiment_with_api/docker_data:/root/docker_data -ti docker_tutorial /bin/bash -c "cd src && source activate ml && python train.py"

3. mount folder with model and start app 
docker run -p 7000:9999 -v /home/user/Desktop/sentiment_with_api/docker_data:/root/docker_data -ti docker_tutorial /bin/bash -c "cd src && source activate ml && python app.py"

4. to get a response from api type in terminal:
curl -X GET "http://127.0.0.1:7000/predict?sentence=amazing%20review"
