async function predictDiabetes() {
    const inputs = document.querySelectorAll("input");

    const features = [
        Number(inputs[0].value),
        Number(inputs[1].value),
        Number(inputs[2].value)
    ];

    const response = await fetch("http://127.0.0.1:5000/predict_diabetes", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ features: features })
    });

    const data = await response.json();

    alert(data.result);
}