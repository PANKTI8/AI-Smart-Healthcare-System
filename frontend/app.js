async function predictDiabetes() {

let glucose = document.getElementById("glucose").value
let bp = document.getElementById("bp").value
let bmi = document.getElementById("bmi").value

let features = [glucose, bp, bmi, 0, 0, 0, 0, 0]

let res = await fetch("http://127.0.0.1:5000/predict_diabetes",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body: JSON.stringify({features:features})
})

let data = await res.json()
document.getElementById("diabetesResult").innerText = data.result

}