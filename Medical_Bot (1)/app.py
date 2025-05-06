from flask import Flask, render_template, request # type: ignore
from huggingface_utils import generate_response

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    user_query = ""
    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            try:
                answer = generate_response(user_query)
            except Exception as e:
                answer = f"Error: {str(e)}"
        else:
            answer = "Please enter a question."
    return render_template("index.html", user_query=user_query, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
