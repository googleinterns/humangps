/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
let model;
let info;
let canvas;
let image1;
let mask1;
let perf;
let pad;
const src_images = [image1, mask1];
const WIDTH = 256;
const HEIGHT = 384;

const state = {
  backend: 'webgl'
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['webgl', 'wasm']).onChange(async backend => {
  await tf.setBackend(backend);
});

function changeImage(input, image_id) {
  if (input.files && input.files.length > 0) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const originalImg = document.getElementById(image_id);
      originalImg.setAttribute('src', e.target.result);
      originalImg.setAttribute('width', WIDTH);
      originalImg.setAttribute('height', HEIGHT);
    };
    if (image_id.indexOf('mask') < 0) {
      loadMask('blank.png', 'im1_mask');
    }
    reader.readAsDataURL(input.files[0]);
  }
};

// Loads an image to the img element.
function loadImage(filename, element_id) {
  let _img = document.getElementById(element_id);
  let newImg = new Image;
  newImg.onload = function() {
    _img.src = this.src;
  }
  newImg.src = 'example_data/' + filename;
}

// Loads a mask into the mask canvas.
function loadMask(filename, element_id) {
  let _mask = document.getElementById(element_id).getContext('2d');
  let _img = new Image;
  _img.onload = function() {
    console.log(_img.width, _img.height, WIDTH, HEIGHT);
    _mask.drawImage(_img, 0, 0, _img.width, _img.height, 0, 0, WIDTH, HEIGHT);
  }
  _img.src = 'example_data/' + filename;
}

function loadPreset(preset_id) {
  loadImage('im' + preset_id + '.jpg', 'im1');
  loadMask('im' + preset_id + '_mask.png', 'im1_mask');
}

function predict() {
  // Tests if the model is loaded.
  if (model == null) {
    alert('Model is not available!');
    return;
  }

  // Tests if an image is missing.
  for (let src_image in src_images) {
    if (src_image.height === 0 || src_image.width === 0) {
      alert('You need to upload an image!');
      return;
    }
  }

  predictButton.textContent = 'Running...';
  predictButton.disabled = true;

  // Sets timeout = 0 to force reload the UI.
  setTimeout(function() {
    const start = Date.now();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    tf.tidy(() => {
      // TODO: Resize images to [384, 256]; tf.image.resize...

      // Normalizes the values from [0, 255] to [-1, 1], same ranged used during training.
      const transform = transformValueRange(0, 255, -1, 1);

      // Converts the input image into a 3D tensor with shape [h, w, colorChannel].
      const image1Tensor = tf.browser.fromPixels(image1);
      const mask1Tensor = tf.browser.fromPixels(mask1);
      const image1Float = tf.cast(image1Tensor, 'float32');
      const mask1Float = tf.cast(mask1Tensor, 'float32');

      const matting1 = tf.sum(mask1Float, /*axis=*/-1, /*keepdims=*/true);
      const threshold = tf.fill(matting1.shape, 150 * 3);
      const mask1Bool = tf.greater(matting1, threshold);

      // Masks the image
      let image1Foreground = tf.where(
        tf.tile(mask1Bool, [1, 1, 3]),
        image1Float,
        tf.zerosLike(image1Float)
      );
      image1Foreground = tf.add(
        tf.mul(image1Foreground, transform.scale),
        transform.offset);

      // Runs the model.
      const input = [
        tf.expandDims(mask1Bool),
        tf.expandDims(image1Foreground)
      ];

      const result = model.predict(input);
      const resultSqueezed = tf.squeeze(result[0]);

      // Adds background.
      const rgbAndBackground = tf.where(
        tf.tile(mask1Bool, [1, 1, 3]),
        resultSqueezed,
        image1Float
      );

      // Renders the result on a canvas.
      const transformBack = transformValueRange(0, 255, 0, 1);

      // Converts back to 0-1.
      const rgbFinal = tf.add(tf.mul(rgbAndBackground, transformBack.scale), transformBack.offset);
      tf.browser.toPixels(rgbFinal, canvas);
    });

    const end = Date.now();
    const time = end - start;
    perf.textContent = time + ' ms';
    predictButton.textContent = 'Process';
    predictButton.disabled = false;
  }, 0);
};

function transformValueRange(
    fromMin, fromMax, toMin, toMax) {
  const fromRange = fromMax - fromMin;
  const ToRange = toMax - toMin;
  const scale = ToRange / fromRange;
  const offset = toMin - fromMin * scale;
  return {scale, offset};
}

function SetPen() {
  pad.minWidth = 3;
  pad.maxWidth = 5;
  pad.penColor = "rgb(255, 255, 255)";
  mask1.classList.add("penpad");
  mask1.classList.remove("eraserpad");
}

function SetBrush() {
  pad.minWidth = 10;
  pad.maxWidth = 15;
  pad.penColor = "rgb(255, 255, 255)";
  mask1.classList.add("penpad");
  mask1.classList.remove("eraserpad");
}

function SetEraser() {
  pad.minWidth = 10;
  pad.maxWidth = 15;
  pad.penColor = "rgb(0, 0, 0)";
  mask1.classList.remove("penpad");
  mask1.classList.add("eraserpad");
}

function ClearBtn() {
  loadMask('blank.png', 'im1_mask');
}

async function setupPage() {
  predictButton = document.getElementById('predict');
  canvas = document.getElementById('result');
  image1 = document.getElementById('im1');
  mask1 = document.getElementById('im1_mask');
  perf = document.getElementById('perf');
  predictButton.textContent = 'Loading...';
  loadMask('im3_mask.png', 'im1_mask');
  if (pad == null) pad = new SignaturePad(mask1);
  SetPen();

  try {
    model = await tf.loadGraphModel('./model/model.json');
  } catch(e)  {
    predictButton.textContent = 'Error in loading model.';
  }
  predictButton.textContent = 'Process';
  predictButton.disabled = false;
};

setupPage();
