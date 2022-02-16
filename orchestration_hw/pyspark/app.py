from flask import Flask, request
from etl import run_etl

app = Flask(__name__)

@app.route('/', methods=['POST'])
def invoke_etl():
    if request.get_json()['task'] != 'etl':
        return {'status': 'undefined request'}
        
    try:
        run_etl()
        return {'status': 'success'}
    except:
        return {'status': 'failure'}
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)