from flask import Flask,render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model= load_model('sudoku_model.h5')
app =Flask(__name__, template_folder= "template")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit",methods=['POST'])
def submit():
    
    
    l=[]
    for i in range(1,10):
        for j in range(1,10):
            k=request.form[f"cell_{i}{j}"]
            if k=="":
                l.append(0)
            else:
                l.append(int(k))
                
    isempty=1
    for i in l:
        if i!=0:
            isempty=0
            break
        
    if isempty==1:
        return render_template("index.html", message="All cell cannot be empty")
    else:
        print(l)
        lnp= np.array(l).reshape(9,9) 
        ln= list(model.predict(lnp.reshape(1,9,9,1)).argmax(-1).squeeze()+1)
        return render_template("solution.html",ans=ln)
