document.addEventListener("DOMContentLoaded", function () {
  const startRecordButton = document.getElementById("startRecord");
  const stopRecordButton = document.getElementById("stopRecord");
  const recordedAudio = document.getElementById("recordedAudio");
  const analyzeButton = document.getElementById("analyze");
  const resultDiv = document.getElementById("result");

  let mediaRecorder;
  let audioChunks = [];
  let recordedBlob;

  // 녹음 시작 버튼 클릭 이벤트 처리
  startRecordButton.addEventListener("click", async () => {
      startRecordButton.disabled = true;
      stopRecordButton.disabled = false;
      audioChunks = [];

      try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = (event) => {
              if (event.data.size > 0) {
                  audioChunks.push(event.data);
              }
          };
          mediaRecorder.onstop = () => {
              recordedBlob = new Blob(audioChunks, { type: "audio/wav" });
              recordedAudio.src = URL.createObjectURL(recordedBlob);
              recordedAudio.style.display = "block";
              analyzeButton.disabled = false;
          };
          mediaRecorder.start();
      } catch (error) {
          console.error("녹음을 시작하는 동안 오류 발생:", error);
      }
  });

  // 녹음 완료 버튼 클릭 이벤트 처리
  stopRecordButton.addEventListener("click", () => {
      startRecordButton.disabled = false;
      stopRecordButton.disabled = true;
      mediaRecorder.stop();
  });

  // 분석 시작 버튼 클릭 이벤트 처리
  analyzeButton.addEventListener("click", async () => {
      if (recordedBlob) {
          const formData = new FormData();
          formData.append("file", recordedBlob);

          try {
              const response = await fetch("/main", {
                  method: "POST",
                  body: formData,
              });
              if (response.ok) {
                  const result = await response.json();
                  console.log("음성 분석 결과:", result);

                  // 결과를 화면에 출력
                  resultDiv.innerHTML = `<p><strong>감정:</strong> ${result.emotion}</p><p><strong>신뢰도:</strong> ${result.confidence}</p>`;
              } else {
                  console.error("오류 발생:", response.status);
              }
          } catch (error) {
              console.error("오류 발생:", error);
          }
      } else {
          alert("음성 파일을 녹음하세요.");
      }
  });
});
