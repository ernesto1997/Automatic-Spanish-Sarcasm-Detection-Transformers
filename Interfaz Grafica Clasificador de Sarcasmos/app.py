from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Cargar el modelo desde Hugging Face
model_checkpoint = "Ernesto-1997/roberta-base-bne-finetuned-spanish_sarcastic_texts"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# Función para predecir la probabilidad de que una frase sea sarcástica
def predict_transformer(frase):
    inputs = tokenizer(frase, return_tensors="pt", padding=True, truncation=True)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilidades = torch.softmax(logits, dim=-1).numpy()
    return probabilidades[0][1]

# Función para generar una frase ingeniosa según la probabilidad
def frase_ingeniosa(probabilidad_sarcasmo):
    if probabilidad_sarcasmo > 0.8:
        return "¡Sarcasmo de nivel maestro, cuidado con ese ingenio afilado!"
    elif 0.5 <= probabilidad_sarcasmo <= 0.8:
        return "Hmm, suena un poco sarcástico, pero también puede ser solo tu humor habitual..."
    else:
        return "No parece sarcasmo. Pero quién sabe, a veces hasta los mejores se equivocan."

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    probabilidad_sarcasmo = None
    frase = None
    frase_ingeniosa_res = None
    if request.method == 'POST':
        frase = request.form['frase']
        probabilidad_sarcasmo = predict_transformer(frase)
        resultado = "Sarcasmo" if probabilidad_sarcasmo >= 0.5 else "No Sarcasmo"
        frase_ingeniosa_res = frase_ingeniosa(probabilidad_sarcasmo)
    return render_template('index.html', frase=frase, resultado=resultado, probabilidad_sarcasmo=probabilidad_sarcasmo, frase_ingeniosa=frase_ingeniosa_res)

if __name__ == '__main__':
    app.run(debug=True)
