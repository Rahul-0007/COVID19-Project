from flask import Flask,render_template,request
app = Flask(__name__,template_folder='template')
import pickle

file = open('model.pkl', 'rb')
clf=pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        fever=int(myDict['fever'])
        age=int(myDict['age'])
        Pain=int(myDict['Pain'])
        runnyNose=int(myDict['runnyNose'])
        diffBreath=int(myDict['diffBreath'])
        #code for inference
        inputfeatures=[fever,Pain,age,runnyNose,diffBreath]
        infection_probability=clf.predict_proba([inputfeatures])[0][1]
        print(infection_probability)
        return render_template('show.html',inf=round(100*infection_probability))
    return render_template('index.html')
        # return 'Hello, World'+ str(infection_probability)

if __name__ == "__main__":
    app.run(debug=True)