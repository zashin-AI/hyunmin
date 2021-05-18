# pip install flask

# 웹 애플리케이션 - 웹을 구성하는 모든 것
# 웹 컨텍스트

# Setup
## index.html : 웹페이지 맨 처음 만들었을 때 나오는 첫 화면

# https://www.youtube.com/watch?v=mqhxxeeTbu0
from flask import Flask, redirect, url_for

app = Flask(__name__)
a = False

@app.route("/")  # 주소 정해주기
def home() :
    return "Hello! This is the main page <h1>HELLO<h1>"

@app.route("/<name>")   # 타이핑한 내용이 들어가진다.
def user(name):
    return f"Hello {name}!"

@app.route("/admin")
def admin() :
    return redirect(url_for("home")) 

if __name__ == "__main__" :
    app.run()
