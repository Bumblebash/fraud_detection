document.getElementById("fraudForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let features = document.getElementById("features").value.split(",").map(Number);

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "features": features })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").textContent = data.prediction === 1 ? "Fraudulent Transaction" : "Legitimate Transaction";
    })
    .catch(error => console.error("Error:", error));
});
