from flask import Flask, render_template, request
from text_summarizatio import summarizer

app = Flask(__name__)

# Define the root route
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define the analyze route
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # Get the raw text input from the form
        rawtext = request.form['rawtext']
        # Summarize the input text
        summary, original_txt, len_orig_txt, len_summary = summarizer(rawtext)
        
        # Render the summary.html template with the summary and text lengths
        return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary)    

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
