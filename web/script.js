// 全局配置
const CONFIG = {
    imgSize: 512,
    latentSize: 64, // 512 / 8
    modelPaths: {
        encoder: './models/encoder.onnx',
        flow:    './models/flow.onnx',
        decoder: './models/decoder.onnx'
    }
};

const SESSIONS = {};
let isEngineReady = false;
let isImageLoaded = false;

// 日志工具
function log(msg, isError = false) {
    const logBox = document.getElementById('logArea');
    const p = document.createElement('p');
    p.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    if (isError) p.style.color = 'red';
    logBox.prepend(p);
    console.log(msg);
}

// 1. 初始化 (排队加载模式)
window.onload = async () => {
    log("🚀 开始初始化...");

    if (!navigator.gpu) {
        log("⚠️ 未检测到 WebGPU，将回退到 CPU (WASM) 模式，速度会很慢。", true);
        alert("建议使用 Chrome 浏览器并开启 WebGPU flags");
    }

    const options = {
        executionProviders: ['webgpu'], // 优先使用 GPU
        enableMemPattern: false,        // 关闭内存优化以提高兼容性
        enableCpuMemArena: false
    };

    try {
        // --- 步骤 1: 加载 Encoder ---
        log("📦 [1/3] 加载 Encoder...");
        SESSIONS.encoder = await ort.InferenceSession.create(CONFIG.modelPaths.encoder, options);
        log("✅ Encoder 就绪");

        // --- 步骤 2: 加载 Flow ---
        log("📦 [2/3] 加载 Flow (核心)...");
        SESSIONS.flow = await ort.InferenceSession.create(CONFIG.modelPaths.flow, options);
        log("✅ Flow 就绪");

        // --- 步骤 3: 加载 Decoder ---
        log("📦 [3/3] 加载 Decoder (解码器)...");
        SESSIONS.decoder = await ort.InferenceSession.create(CONFIG.modelPaths.decoder, options);
        log("✅ Decoder 就绪");

        // 全部完成
        isEngineReady = true;
        document.getElementById('engineStatus').innerText = "🟢 引擎就绪 (WebGPU)";
        document.getElementById('engineStatus').style.background = "#e8f5e9";
        document.getElementById('engineStatus').style.color = "#2e7d32";
        log("🎉 所有模型加载完成！请上传图片。");
        
        checkButtonState();

    } catch (e) {
        log(`❌ 初始化失败: ${e.message}`, true);
        console.error(e);
        document.getElementById('runBtn').innerText = "❌ 加载出错 (看日志)";
    }
};

// 2. UI 交互
const fileInput = document.getElementById('fileInput');
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    document.getElementById('previewImg').src = url;
    document.getElementById('previewImg').style.display = 'block';
    document.getElementById('placeholder').style.display = 'none';

    // 预加载到 Canvas
    const img = new Image();
    img.onload = () => {
        const ctx = document.getElementById('processCanvas').getContext('2d');
        ctx.drawImage(img, 0, 0, CONFIG.imgSize, CONFIG.imgSize);
        isImageLoaded = true;
        log("📷 图片已加载");
        checkButtonState();
    };
    img.src = url;
};

function checkButtonState() {
    const btn = document.getElementById('runBtn');
    if (isEngineReady && isImageLoaded) {
        btn.disabled = false;
        btn.innerText = "🚀 开始生成 (Start)";
        btn.onclick = runPipeline;
    } else if (isEngineReady && !isImageLoaded) {
        btn.innerText = "👈 请先上传图片";
    }
}

// 3. 推理管线 (核心)
async function runPipeline() {
    const btn = document.getElementById('runBtn');
    btn.disabled = true;
    btn.innerText = "🔄 生成中...";
    document.getElementById('resultArea').style.display = 'block';

    try {
        const steps = parseInt(document.getElementById('stepRange').value);
        const styleId = parseInt(document.getElementById('styleSelect').value);

        // --- Phase 1: Encode ---
        log("🔄 正在编码 (Encoder)...");
        // 强制 UI 刷新
        await new Promise(r => setTimeout(r, 20)); 
        
        const inputTensor = preprocess();
        const encOut = await SESSIONS.encoder.run({ input: inputTensor });
        const x_cond = encOut.output;

        // --- Phase 2: Flow Loop ---
        log("🌊 开始 Flow 采样...");
        let x_t = createGaussianNoise(1, 4, CONFIG.latentSize, CONFIG.latentSize);
        const dt = 1.0 / steps;

        for (let i = 0; i < steps; i++) {
            // 更新进度条
            const progress = Math.round((i / steps) * 100);
            document.getElementById('progressBar').style.width = `${progress}%`;
            
            // 构造标量输入
            const tTensor = new ort.Tensor('float32', new Float32Array([i / steps]), [1]);
            const sTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(styleId)]), [1]);

            // 执行 Flow
            // 注意：这里的 key 必须和 Python 导出时的 input_names 一致
            const feeds = { x_t: x_t, x_cond: x_cond, t: tTensor, s: sTensor };
            const results = await SESSIONS.flow.run(feeds);
            const v_pred = results.output;

            // Euler 更新
            x_t = eulerUpdate(x_t, v_pred, dt);

            // 让出主线程，防止页面卡死
            await new Promise(r => requestAnimationFrame(r));
        }

        // --- Phase 3: Decode ---
        log("🎨 正在解码 (Decoder)...");
        const decOut = await SESSIONS.decoder.run({ input: x_t });

        // --- Phase 4: Display ---
        postprocess(decOut.output);
        document.getElementById('progressBar').style.width = "100%";
        log("✨ 生成完成！");
        btn.innerText = "✨ 再来一张";

    } catch (e) {
        log(`❌ 运行错误: ${e.message}`, true);
        console.error(e);
    } finally {
        btn.disabled = false;
    }
}

// === 数学工具 ===

function eulerUpdate(x, v, dt) {
    const xData = x.data;
    const vData = v.data;
    const newData = new Float32Array(xData.length);
    for (let i = 0; i < xData.length; i++) {
        newData[i] = xData[i] + vData[i] * dt;
    }
    return new ort.Tensor('float32', newData, x.dims);
}

function createGaussianNoise(b, c, h, w) {
    const size = b * c * h * w;
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        // 简单的高斯近似 (Box-Muller 虽然准但在 JS 里为了速度可以用 Uniform 近似或者手写 Box-Muller)
        // 这里用 Box-Muller 保证质量
        const u = 1 - Math.random();
        const v = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        data[i] = z;
    }
    return new ort.Tensor('float32', data, [b, c, h, w]);
}

// === 图像处理 ===

function preprocess() {
    const ctx = document.getElementById('processCanvas').getContext('2d');
    const imageData = ctx.getImageData(0, 0, CONFIG.imgSize, CONFIG.imgSize);
    const { data } = imageData;
    const floatArr = new Float32Array(3 * CONFIG.imgSize * CONFIG.imgSize);

    // HWC -> CHW, Normalize [-1, 1]
    for (let i = 0; i < CONFIG.imgSize * CONFIG.imgSize; i++) {
        floatArr[i] = (data[i * 4] / 255.0 - 0.5) / 0.5; // R
        floatArr[i + CONFIG.imgSize * CONFIG.imgSize] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5; // G
        floatArr[i + 2 * CONFIG.imgSize * CONFIG.imgSize] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5; // B
    }
    return new ort.Tensor('float32', floatArr, [1, 3, CONFIG.imgSize, CONFIG.imgSize]);
}

function postprocess(tensor) {
    const data = tensor.data;
    const canvas = document.getElementById('outputCanvas');
    canvas.width = CONFIG.imgSize;
    canvas.height = CONFIG.imgSize;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(CONFIG.imgSize, CONFIG.imgSize);
    
    // CHW -> HWC, Denormalize
    for (let i = 0; i < CONFIG.imgSize * CONFIG.imgSize; i++) {
        // Clamp to [0, 255]
        const r = Math.min(255, Math.max(0, (data[i] * 0.5 + 0.5) * 255));
        const g = Math.min(255, Math.max(0, (data[i + CONFIG.imgSize * CONFIG.imgSize] * 0.5 + 0.5) * 255));
        const b = Math.min(255, Math.max(0, (data[i + 2 * CONFIG.imgSize * CONFIG.imgSize] * 0.5 + 0.5) * 255));
        
        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
}