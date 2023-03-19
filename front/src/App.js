import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const CanvasComponent = () => {
  const [isPainting, setIsPainting] = useState(false);
  const canvasRef = useRef(null);
  const contextRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.width = '280px';
    canvas.style.height = '280px';
    canvas.style.backgroundColor = 'black';

    const context = canvas.getContext('2d');
    context.scale(0.1, 0.1);
    context.lineCap = 'round';
    context.strokeStyle = 'white';
    context.lineWidth = 1;
    contextRef.current = context;
  }, []);

  const startPaint = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.beginPath();
    contextRef.current.moveTo(offsetX, offsetY);
    setIsPainting(true);
  };

  const endPaint = () => {
    contextRef.current.closePath();
    setIsPainting(false);
  };

  const paint = ({ nativeEvent }) => {
    if (!isPainting) return;

    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.lineTo(offsetX, offsetY);
    contextRef.current.stroke();
  };

  const clearCanvas = () => {
    contextRef.current.clearRect(0, 0, 280, 280);
  }

  const [predictionResult, setPredictionResult] = useState("");

  const handleClick = async () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const imgData = context.getImageData(0, 0, canvas.width, canvas.height);
    const data = [];
    for (let i = 0; i < imgData.data.length; i += 4) {
      data.push(imgData.data[i])
    }
    await axios.post('/predict', {
      image: data,
    }).then(response => {
      setPredictionResult(response.data.prediction)
    })
  };

  return (
    <div>
      <canvas 
        ref={canvasRef} 
        onMouseDown={startPaint}
        onMouseUp={endPaint}
        onMouseMove={paint} />

      <br/><br/>

      <button onClick={() => clearCanvas()}>Clear</button>&nbsp;&nbsp;
      <button onClick={() => handleClick()}>Predict</button>

      <br/><br/>

      <label>Prediction:</label>&nbsp;&nbsp;
      <span>{predictionResult}</span>
    </div>
  );
}

function App() {
  return (
    <div className="App">
      <h1>Handwritten Digit Recognition</h1>
      <CanvasComponent/>
    </div>
  );
}

export default App;
