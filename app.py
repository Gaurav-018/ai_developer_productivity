import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Safely load your AdaBoost pickle file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Adaboost_pkl.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# The precise feature list expected by your model
FEATURE_NAMES = [
    'Hours_Coding', 'AI_Usage_Hours', 'Lines_of_Code', 'Commits',
    'Bugs_Reported', 'Sleep_Hours', 'Distractions', 'Cognitive_Load', 'Stress_Level'
]

# The complete colorful frontend embedded directly in the script
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Developer Analytics Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    </style>
</head>
<body class="text-slate-100 min-h-screen flex items-center justify-center p-4 md:p-8">

    <div class="max-w-4xl w-full bg-slate-900/60 backdrop-blur-xl rounded-3xl border border-slate-800 shadow-2xl overflow-hidden grid grid-cols-1 md:grid-cols-5">
        
        <div class="md:col-span-2 bg-gradient-to-br from-violet-600 via-indigo-700 to-cyan-500 p-8 flex flex-col justify-between text-white">
            <div class="space-y-4">
                <span class="bg-white/20 text-xs font-semibold uppercase tracking-wider px-3 py-1 rounded-full backdrop-blur-md inline-block">Adaptive AI Engine</span>
                <h1 class="text-3xl font-extrabold tracking-tight">AdaBoost Classifier</h1>
                <p class="text-indigo-100 text-sm leading-relaxed">Enter your development metrics to evaluate performance profiles, optimization bottlenecks, and target outputs instantly.</p>
            </div>
            <div class="pt-8 text-xs text-indigo-200/80">Data Model Pipeline v1.6.1</div>
        </div>

        <form id="predictionForm" class="md:col-span-3 p-8 space-y-6 bg-slate-900/40">
            <h2 class="text-xl font-bold text-slate-200 border-b border-slate-800 pb-3">Operational Feature Space</h2>
            
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {% for feature in features %}
                <div class="space-y-1.5">
                    <label for="{{ feature }}" class="block text-xs font-medium text-slate-400 tracking-wide">
                        {{ feature.replace('_', ' ') }}
                    </label>
                    <input 
                        type="number" 
                        step="any" 
                        name="{{ feature }}" 
                        id="{{ feature }}" 
                        placeholder="0.00"
                        required 
                        class="w-full bg-slate-950/50 border border-slate-800 text-slate-100 placeholder-slate-600 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500 transition-all shadow-inner"
                    >
                </div>
                {% endfor %}
            </div>

            <div class="pt-4">
                <button type="submit" class="w-full bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-600 hover:to-cyan-600 text-white font-semibold py-3 px-6 rounded-xl shadow-lg shadow-violet-500/20 transform active:scale-[0.99] transition-all text-sm tracking-wide uppercase">
                    Compute Classification Profile
                </button>
            </div>
        </form>
    </div>

    <div id="resultModal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-md hidden flex items-center justify-center p-4 z-50">
        <div class="bg-slate-900 border border-slate-800 max-w-md w-full rounded-3xl p-8 shadow-2xl text-center space-y-6">
            <div class="mx-auto w-16 h-16 bg-emerald-500/10 text-emerald-400 rounded-full flex items-center justify-center border border-emerald-500/20 shadow-inner">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            </div>
            <div class="space-y-2">
                <h3 class="text-2xl font-bold text-slate-200">Evaluation Result</h3>
                <p class="text-sm text-slate-400">The prediction model resolved to state target:</p>
            </div>
            <div class="bg-slate-950/80 rounded-2xl p-4 border border-slate-800 shadow-inner">
                <span id="predictionResult" class="text-4xl font-extrabold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">--</span>
            </div>
            <button type="button" onclick="closeModal()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium py-2.5 px-5 rounded-xl transition-colors text-sm">
                Dismiss Analytics Window
            </button>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if(data.success) {
                    document.getElementById('predictionResult').innerText = data.prediction;
                    document.getElementById('resultModal').classList.remove('hidden');
                } else {
                    alert('Calculation Error: ' + data.error);
                }
            } catch (err) {
                alert('Network execution failed.');
            }
        });

        function closeModal() {
            document.getElementById('resultModal').classList.add('hidden');
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    # Render the markup directly out of the string structure
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feature in FEATURE_NAMES:
            val = float(request.form.get(feature, 0.0))
            input_data.append(val)
        
        final_features = np.array([input_data])
        prediction = int(model.predict(final_features)[0])
        
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
