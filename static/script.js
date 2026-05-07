const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

ctx.strokeStyle = "white";
ctx.lineWidth = 15;
ctx.lineCap = "round";

canvas.addEventListener("mousedown", (e) => {

    drawing = true;

    ctx.beginPath();

    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
});

canvas.addEventListener("mouseleave", () => {
    drawing = false;
});

canvas.addEventListener("mousemove", draw);

function draw(e) {

    if (!drawing) return;

    ctx.lineTo(e.offsetX, e.offsetY);

    ctx.stroke();
}

async function predict() {

    const image = canvas.toDataURL();

    const response = await fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            image: image
        })
    });

    const result = await response.json();

    document.getElementById("resultado")
        .innerText =
        "Número: " + result.prediction +
        " | Confianza: " + result.confidence + "%";
}

function clearCanvas() {

    ctx.fillStyle = "black";

    ctx.fillRect(0, 0, canvas.width, canvas.height);

    document.getElementById("resultado")
        .innerText = "";
}