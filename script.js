// 🎯 Smooth Scroll to Predict Section
function scrollToPredict() {
    document.getElementById("predict").scrollIntoView({ behavior: "smooth" });
}


// 🎯 Scroll Reveal Animation
window.addEventListener("scroll", () => {
    document.querySelectorAll(".reveal").forEach(el => {
        const windowHeight = window.innerHeight;
        const elementTop = el.getBoundingClientRect().top;

        if (elementTop < windowHeight - 100) {
            el.classList.add("active");
        }
    });
});


// 🎯 Prediction Flow (Loading → Result)
async function predictMovie() {
    const loading = document.querySelector(".loading");
    const result = document.querySelector(".result");

    // Get values
    const director = document.getElementById("director").value;
    const actors = document.getElementById("actors").value;
    const genre = document.getElementById("genre").value;
    const budget = document.getElementById("budget").value;

    // ✅ ADD HERE
    if (!director || !actors || !genre || !budget) {
        alert("Please fill all fields");
        return;
    }

    // Show loading
    loading.style.display = "flex";
    result.classList.remove("show");

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                director: director,
                actors: actors,
                genre: genre,
                budget: budget
            })
        });

        const data = await response.json();

        loading.style.display = "none";

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        showResult(data.Prediction, data.Confidence);
        result.classList.add("show");

    } catch (error) {
        loading.style.display = "none";
        alert("API connection failed");
        console.error(error);
    }
}

// ML Model Predicted Result
function showResult(prediction, confidence) {

    const title = document.querySelector(".result-title");

    // Set label
    title.innerText = prediction.toUpperCase();

    // Set color
    if (prediction.toLowerCase() === "hit") {
        title.className = "result-title result-hit";
    } else if (prediction.toLowerCase() === "average") {
        title.className = "result-title result-average";
    } else {
        title.className = "result-title result-flop";
    }

    // Set confidence
    document.querySelector(".meter-fill").innerText = confidence + "%";
}