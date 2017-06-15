import os
from flask import Flask, render_template, request

__author__ = 'paul'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/", methods=["GET", "POST"])
def upload():
	msg = ""
	destination = ""
	target = os.path.join(APP_ROOT, "videos/")

	if not os.path.isdir(target):
		os.mkdir(target)

	for file in request.files.getlist("video"):
		filename = file.filename
		destination = os.path.join(target, filename)
		file.save(destination)
	
	print(os.path.isfile(destination))


	if os.path.isfile(destination):
		msg = "Upload Complete"
		print(msg)
	return render_template('upload.html', msg=msg)

if __name__ == "__main__":
	app.run(debug=True)