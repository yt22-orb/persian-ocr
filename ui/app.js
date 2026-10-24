(() => {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ALPHABET = "ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی".split(" ");
  const screens = [...document.querySelectorAll("[data-screen]")];
  const camera = document.querySelector("#camera");
  const cameraPlaceholder = document.querySelector("#camera-placeholder");
  const captureButton = document.querySelector("#capture-button");
  const captureCanvas = document.querySelector("#capture-canvas");
  const dropZone = document.querySelector("#drop-zone");
  const fileInput = document.querySelector("#file-input");
  const uploadThumbnail = document.querySelector("#upload-thumbnail");
  const fileName = document.querySelector("#file-name");
  const previewImage = document.querySelector("#preview-image");
  const processingModal = document.querySelector("#processing-modal");
  const errorModal = document.querySelector("#error-modal");
  const errorTitle = document.querySelector("#error-title");
  const errorMessage = document.querySelector("#error-message");
  const resultLetter = document.querySelector("#result-letter");
  const resultName = document.querySelector("#result-name");
  const resultConfidence = document.querySelector("#result-confidence");
  const toast = document.querySelector("#toast");
  const resultImage = document.querySelector("#result-image");
  const matchesList = document.querySelector("#matches-list");
  const alphabetGrid = document.querySelector("#alphabet-grid");
 
  let activeScreen = "home";
  let source = null;
  let selectedImage = null;
  let selectedImageUrl = null;
  let cameraStream = null;
  let lastFocusedElement = null;
  let toastTimer = null;
  let scanInProgress = false;

  function showScreen(name) {
    activeScreen = name;
    screens.forEach((screen) => {
      const isActive = screen.dataset.screen === name;
      screen.hidden = !isActive;
      screen.classList.toggle("screen--active", isActive);
    });
    if (name !== "home") stopCamera();
    const heading = document.querySelector(`[data-screen="${name}"] h1`);
    heading?.focus?.({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function setTab(name) {
    document.querySelectorAll(".tab").forEach((tab) => {
      const on = tab.dataset.action === `open-${name}`;
      tab.classList.toggle("tab--active", on);
      tab.setAttribute("aria-selected", String(on));
    });
    document.querySelectorAll(".pane").forEach((pane) => { pane.hidden = pane.id !== `${name}-pane`; });
  }

  function closeAlphabet() {
    document.body.classList.remove("alphabet-open");
  }

  function setImage(blob, origin) {
    if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
    selectedImage = blob;
    selectedImageUrl = URL.createObjectURL(blob);
    source = origin;
    previewImage.src = selectedImageUrl;
    previewImage.classList.remove("is-flipped");
  }

  function resetUpload() {
    fileInput.value = "";
    fileName.textContent = "No image selected";
    uploadThumbnail.innerHTML = '<svg viewBox="0 0 64 64"><rect x="10" y="10" width="44" height="44" rx="5"/><circle cx="23" cy="24" r="4"/><path d="m14 47 12-12 8 8 7-7 9 11"/></svg>';
  }

  function resetAll(keepCamera = false) {
    if (!keepCamera) stopCamera();
    if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
    selectedImage = null;
    selectedImageUrl = null;
    source = null;
    previewImage.removeAttribute("src");
    resetUpload();
  }

  async function openCamera() {
    source = "capture";
    setTab("camera");
    if (cameraStream) return;
    captureButton.disabled = true;
    cameraPlaceholder.hidden = false;
    cameraPlaceholder.querySelector("span").textContent = "Starting camera…";

    if (!navigator.mediaDevices?.getUserMedia) {
      showError("Camera unavailable", "This browser cannot access a camera. Upload an image instead.");
      return;
    }

    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 1280 } },
        audio: false,
      });
      camera.srcObject = cameraStream;
      await camera.play();
      cameraPlaceholder.hidden = true;
      captureButton.disabled = false;
    } catch (error) {
      cameraPlaceholder.querySelector("span").textContent = "Camera could not start";
      const denied = error?.name === "NotAllowedError";
      showError(
        denied ? "Camera permission needed" : "Camera unavailable",
        denied ? "Allow camera access in your browser, then try again." : "We could not start the camera. Upload an image instead."
      );
    }
  }

  function stopCamera() {
    if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    camera.srcObject = null;
    captureButton.disabled = true;
  }

  function capturePhoto() {
    if (!camera.videoWidth || !camera.videoHeight) {
      showError("Camera is not ready", "Wait for the camera preview, then try again.");
      return;
    }
    captureCanvas.width = camera.videoWidth;
    captureCanvas.height = camera.videoHeight;
    const context = captureCanvas.getContext("2d");
    context.drawImage(camera, 0, 0);
    captureCanvas.toBlob((blob) => {
      if (!blob) {
        showError("Photo could not be saved", "Please capture the image again.");
        return;
      }
      setImage(blob, "capture");
    }, "image/jpeg", 0.92);
  }

  function validateFile(file) {
    if (!file) return "Choose a PNG or JPG image first.";
    if (!["image/png", "image/jpeg"].includes(file.type)) return "Only PNG and JPG images are supported.";
    if (file.size > MAX_BYTES) return "The image must be 10 MB or smaller.";
    return null;
  }

  async function useUpload(file) {
    const validationError = validateFile(file);
    if (validationError) {
      showError("Invalid image", validationError);
      return;
    }

    try {
      await verifyImage(file);
      setImage(file, "upload");
      fileName.textContent = file.name;
      uploadThumbnail.innerHTML = "";
      const thumbnail = document.createElement("img");
      thumbnail.src = selectedImageUrl;
      thumbnail.alt = "";
      uploadThumbnail.append(thumbnail);
    } catch {
      showError("Invalid image", "This file could not be read as a PNG or JPG image.");
    }
  }

  function verifyImage(file) {
    return new Promise((resolve, reject) => {
      const testImage = new Image();
      const url = URL.createObjectURL(file);
      testImage.onload = () => {
        URL.revokeObjectURL(url);
        if (Math.min(testImage.naturalWidth, testImage.naturalHeight) < 16) reject(new Error("too small"));
        else resolve();
      };
      testImage.onerror = () => { URL.revokeObjectURL(url); reject(new Error("decode failed")); };
      testImage.src = url;
    });
  }

  function showProcessing(show) {
    processingModal.hidden = !show;
    if (show) document.body.setAttribute("aria-busy", "true");
    else document.body.removeAttribute("aria-busy");
  }

  function showError(title, message, mode = "close") {
    lastFocusedElement = document.activeElement;
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorModal.dataset.mode = mode;
    errorModal.hidden = false;
    errorModal.querySelector(".modal-close").focus();
  }

  function closeError() {
    errorModal.hidden = true;
    lastFocusedElement?.focus?.();
  }

  function handleApiError(status, payload) {
    if (status === 422 || payload?.code === "unrecognized") {
      showError("Could not recognize letter", payload?.message || "Please try again with a clearer picture.", "retake");
      return;
    }
    if (status === 503 || payload?.code === "model_unavailable") {
      showError("Model can’t load", payload?.message || "The model is not available right now.", "retry");
      return;
    }
    showError("Scan failed", payload?.message || "Something went wrong while scanning. Please try again.", "retry");
  }

  async function scanImage() {
    if (scanInProgress) return;
    if (!selectedImage) {
      showError("No image selected", "Take or upload an image before scanning.");
      return;
    }

    scanInProgress = true;
    showProcessing(true);
    const started = performance.now();
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 30000);
    try {
      const image = await getScanBlob();
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": image.type || "image/jpeg" },
        body: image,
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      const remaining = Math.max(0, 850 - (performance.now() - started));
      await new Promise((resolve) => window.setTimeout(resolve, remaining));
      showProcessing(false);

      if (!response.ok || !payload.accepted) {
        handleApiError(response.status, payload);
        return;
      }
      showResult(payload.matches || [payload]);
    } catch (error) {
      showProcessing(false);
      const timedOut = error?.name === "AbortError";
      showError(
        timedOut ? "Scan timed out" : "Model can’t load",
        timedOut ? "The model took too long to respond. Please try again." : "Start the UI with serve_ui.py, then try again.", "retry"
      );
    } finally {
      window.clearTimeout(timeout);
      scanInProgress = false;
    }
  }

  async function getScanBlob() {
    if (!previewImage.classList.contains("is-flipped")) return selectedImage;
    const bitmap = await createImageBitmap(selectedImage);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(bitmap, 0, 0);
    const type = selectedImage.type === "image/png" ? "image/png" : "image/jpeg";
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob || selectedImage), type, 0.92));
  }

  function showResult(matches) {
    const [main] = matches;
    const percent = (confidence) => confidence > 0 && confidence < 0.0005
      ? "<0.1%"
      : `${(confidence * 100).toFixed(1)}%`;
    resultLetter.textContent = main.letter;
    resultName.textContent = main.transliteration;
    resultConfidence.textContent = percent(main.confidence);
    resultConfidence.parentElement.style.setProperty("--value", percent(main.confidence));
    resultImage.src = selectedImageUrl;
    resultImage.classList.toggle("is-flipped", previewImage.classList.contains("is-flipped"));
    matchesList.innerHTML = matches.slice(0, 4).map((match) => `<li class="match"><span class="match__letter" lang="fa" dir="rtl">${match.letter}</span><span class="match__name">${match.transliteration}</span><span class="match__score">${percent(match.confidence)}</span></li>`).join("");
    alphabetGrid.querySelectorAll(".alphabet-tile").forEach((tile) => {
      tile.classList.toggle("alphabet-tile--main", tile.textContent === main.letter);
      tile.classList.toggle("alphabet-tile--match", matches.slice(1).some((match) => match.letter === tile.textContent));
    });
    showScreen("result");
  }

  async function copyResult() {
    const text = resultLetter.textContent;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    showToast("Letter copied");
  }

  function showToast(message) {
    window.clearTimeout(toastTimer);
    toast.textContent = message;
    toast.hidden = false;
    toastTimer = window.setTimeout(() => { toast.hidden = true; }, 1800);
  }

  function retake() {
    const previousSource = source;
    closeError();
    resetAll();
    showScreen("home");
    if (previousSource === "upload") setTab("upload");
    else openCamera();
  }

  document.addEventListener("click", (event) => {
    const control = event.target.closest("[data-action]");
    if (!control) return;
    const actions = {
      "open-camera": openCamera,
      "open-upload": () => { stopCamera(); resetUpload(); setTab("upload"); },
      home: () => { closeError(); closeAlphabet(); resetAll(); setTab("camera"); showScreen("home"); },
      capture: capturePhoto,
      flip: () => previewImage.classList.toggle("is-flipped"),
      clear: () => resetAll(true),
      retake,
      retry: () => { closeError(); scanImage(); },
      scan: scanImage,
      copy: copyResult,
      "open-alphabet": () => document.body.classList.add("alphabet-open"),
      "close-alphabet": closeAlphabet,
      "close-error": closeError,
    };
    actions[control.dataset.action]?.();
  });

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (event) => {
    if (["Enter", " "].includes(event.key)) { event.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => useUpload(fileInput.files?.[0]));

  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("drop-zone--dragging");
  }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drop-zone--dragging");
  }));
  dropZone.addEventListener("drop", (event) => useUpload(event.dataTransfer?.files?.[0]));

  errorModal.addEventListener("click", (event) => { if (event.target === errorModal) closeError(); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !errorModal.hidden) closeError();
    if (event.key === "Escape") closeAlphabet();
    if (event.key === "Tab" && !errorModal.hidden) {
      const focusable = [...errorModal.querySelectorAll("button:not(:disabled)")].filter((el) => el.offsetParent !== null);
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });

  alphabetGrid.innerHTML = ALPHABET.map((letter) => `<li class="alphabet-tile" lang="fa">${letter}</li>`).join("");
  
  window.addEventListener("beforeunload", () => {
    stopCamera();
    if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  });
})();
