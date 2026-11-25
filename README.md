# Stock Lab - swing trading web app

## Run on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
Your browser opens at http://localhost:8501.
Turn on "Demo data" in the sidebar to practise without internet.

## Put it online (free)
1. Push app.py, stock_lab.py and requirements.txt to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, click "Create app", pick the repo and set the main file to app.py.
3. You get a public link that opens on your phone or any browser.
   (Railway also works: start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.)

Alpaca keys: type them on the Paper trade tab. They are never saved.
Educational tool only - not financial advice.
