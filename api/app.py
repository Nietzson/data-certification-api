
from fastapi import FastAPI
import pandas as pd
from exampack.trainer import Trainer
from joblib import load

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness, danceability, duration_ms, energy, explicit, id, instrumentalness, key, liveness, loudness, mode, name, release_date, speechiness, tempo, valence, artist):
    
    # dico = {
    #     "acousticness": acousticness,
    #     "danceability": danceability ,
    #     "duration_ms" : duration_ms,
    #     "energy": energy,
    #      "explicit" : explicit,
    #      "id" : id, 
    #      "instrumentalness" : instrumentalness,
    #      "key" : key,
    #      "liveness" : liveness,
    #      "loudness" : loudness,
    #      "mode" : mode,
    #      "name" : name,
    #      "release_date" : release_date,
    #      "speechiness" : speechiness,
    #      "tempo" : tempo,
    #      "valence" : valence,
    #      "artist" : artist
    # }
    
    tmp = [float(acousticness), float(danceability), int(duration_ms), float(energy), int(explicit), str(id), float(instrumentalness), int(key), float(liveness), float(loudness), int(mode), str(name), str(release_date), float(speechiness), float(tempo), float(valence), str(artist)]
    tmp_df = pd.DataFrame(tmp)
    tmp_df = tmp_df.T
    tmp_df.rename(columns = {
        0:"acousticness",
        1:"danceability",
        2:"duration_ms" ,
        3:"energy",
         4:"explicit",
         5:"id", 
         6:"instrumentalness",
         7:"key",
         8:"liveness",
         9:"loudness",
         10:"mode" ,
         11:"name" ,
         12:"release_date" ,
         13:"speechiness" ,
         14:"tempo",
         15:"valence",
         16:"artist" 
    }, inplace = True)
    
    model = load("model.joblib")
    
    prediction = model.predict(tmp_df)

    
    return {
        "artist": artist,
        "name": name,
        "popularity": prediction[0]}
        