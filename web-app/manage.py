import base64
from io import BytesIO
import os
from flask import Flask, request, render_template, redirect, url_for
from flask_dropzone import Dropzone
from dog_breeds_detector import plot_img
import shutil

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=30,
)

dropzone = Dropzone(app)


@app.route("/predict")
def predict():
    try:
        fig = plot_img()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        shutil.rmtree("uploads")
        os.mkdir("uploads")
    except Exception as e:
        return "<h2> Please upload image first! </h2>"
    return f"<img src='data:image/png;base64,{data}'/>"


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files.get('file')
            file_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
            f.save(file_path)
    if request.method == 'GET':
        if request.args.get('submit_button') == 'Predict dog breeds':
            return redirect(url_for('predict'), code=302)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
