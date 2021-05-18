# https://www.youtube.com/watch?v=dam0GPOAvVI

from website import create_app

app = create_app()

if __name__ == '__main__' : # 해당 파일을 실행시킬 때만 수행된다.
    app.run(debug=True)

