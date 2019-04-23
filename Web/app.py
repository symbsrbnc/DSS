from flask import Flask, render_template

app=Flask(__name__)


@app.route('/')
def home():
     return render_template('mainpage.html')

@app.route('/inputs')
def inputs():
     return render_template('inputs.html')

@app.route('/result')
def result():
     return render_template('result.html')

@app.route('/search')
def search():
     return render_template('searchpage.html')

if __name__=='__main__':
    app.run(port=8080, debug=True)

