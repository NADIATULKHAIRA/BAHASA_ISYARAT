function updatePrediction(){

    fetch('/prediction')

    .then(response => response.json())

    .then(data => {

        document.getElementById("result").innerText =
        data.result;


        document.getElementById("confidence").innerText =
        data.confidence + "%";


        document.getElementById("bar").style.width =
        data.confidence + "%";

    });

}


setInterval(updatePrediction,300);