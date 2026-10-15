from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, input_size, is_output_linear=False):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.is_output_linear = is_output_linear

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        if self.is_output_linear:
            return total
        return sigmoid(total)


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, is_regression=False):
        self.is_regression = is_regression
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size, is_output_linear=is_regression) for _ in range(output_size)]

    def feedforward(self, inputs):
        hidden_outputs = np.array([neuron.feedforward(inputs) for neuron in self.hidden_layer])
        outputs = np.array([neuron.feedforward(hidden_outputs) for neuron in self.output_layer])
        return outputs


def load_model(json_path, is_regression):
    with open(json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    input_size = len(saved["hidden"][0]["weights"])
    hidden_size = len(saved["hidden"])
    output_size = len(saved["output"])
    net = SimpleNeuralNetwork(input_size, hidden_size, output_size, is_regression)
    for neuron, params in zip(net.hidden_layer, saved["hidden"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]
    for neuron, params in zip(net.output_layer, saved["output"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]
    return net, saved


net_cls, saved_cls = load_model("cat_breed_weights.json", is_regression=False)
net_reg, saved_reg = load_model("cat_weight_weights.json", is_regression=True)


def encode_category(value, classes_list):
    try:
        return classes_list.index(value)
    except ValueError:
        return 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/classify", response_class=HTMLResponse)
async def classify_get(request: Request):
    return templates.TemplateResponse(request=request, name="classify.html")


@app.post("/classify", response_class=HTMLResponse)
async def classify_post(
        request: Request,
        age: float = Form(...),
        weight: float = Form(...),
        body_length: float = Form(...),
        gender: str = Form(...),
        color: str = Form(...)
):
    gender_enc = encode_category(gender, saved_cls["le_gender"])
    color_enc = encode_category(color, saved_cls["le_color"])
    x = np.array([age, weight, body_length, gender_enc, color_enc])

    mn, mx = np.array(saved_cls["mn"]), np.array(saved_cls["mx"])
    x_norm = (x - mn) / (mx - mn)

    preds = net_cls.feedforward(x_norm)
    class_idx = int(np.argmax(preds))
    result = saved_cls["names"][class_idx]

    return templates.TemplateResponse(
        request=request, name="result.html",
        context={"task": "Классификация породы", "result": result}
    )


@app.get("/regress", response_class=HTMLResponse)
async def regress_get(request: Request):
    return templates.TemplateResponse(request=request, name="regress.html")


@app.post("/regress", response_class=HTMLResponse)
async def regress_post(
        request: Request,
        age: float = Form(...),
        body_length: float = Form(...),
        breed: str = Form(...),
        gender: str = Form(...),
        color: str = Form(...)
):
    breed_enc = encode_category(breed, saved_reg["le_breed"])
    gender_enc = encode_category(gender, saved_reg["le_gender"])
    color_enc = encode_category(color, saved_reg["le_color"])
    x = np.array([age, body_length, breed_enc, gender_enc, color_enc])

    mn, mx = np.array(saved_reg["mn"]), np.array(saved_reg["mx"])
    x_norm = (x - mn) / (mx - mn)

    pred_norm = net_reg.feedforward(x_norm)[0]
    pred = pred_norm * (saved_reg["mx_y"] - saved_reg["mn_y"]) + saved_reg["mn_y"]
    result = f"{pred:.2f} кг"

    return templates.TemplateResponse(
        request=request, name="result.html",
        context={"task": "Регрессия (вес кота)", "result": result}
    )