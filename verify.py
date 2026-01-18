import MNN
import cv2
import numpy as np
import os

# ================= 配置 =================
MODEL_DIR = "./mnn_export_final"
IMG_PATH = "./test.jpg"  # 你的测试图
# =======================================

def run():
    print(">>> MNN 最终验证 (复刻 inf.py 逻辑) <<<")
    
    # 1. 加载
    interp_enc = MNN.Interpreter(f"{MODEL_DIR}/Encoder.mnn")
    interp_flow = MNN.Interpreter(f"{MODEL_DIR}/Flow.mnn")
    interp_dec = MNN.Interpreter(f"{MODEL_DIR}/Decoder.mnn")
    
    sess_enc = interp_enc.createSession()
    sess_flow = interp_flow.createSession()
    sess_dec = interp_dec.createSession()
    
    # 2. 预处理 (与 inf.py 一致: Normalize 0.5)
    img = cv2.imread(IMG_PATH)
    if img is None:
        print("❌ 找不到图片")
        return
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 公式: (x - 0.5)/0.5 = 2x - 1 = (x/127.5) - 1.0
    img_input = (img.astype(np.float32) / 127.5) - 1.0
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, :] # NCHW
    # MNN Python 接口极其挑剔，必须转为连续内存
    img_input = np.ascontiguousarray(img_input)
    
    # 3. Encoder
    print("[1/3] Encoder...")
    in_tensor = interp_enc.getSessionInput(sess_enc, "input")
    # 创建临时 Tensor 包装 numpy 数据
    tmp_in = MNN.Tensor((1, 3, 512, 512), MNN.Halide_Type_Float, img_input, MNN.Tensor_DimensionType_Caffe)
    in_tensor.copyFrom(tmp_in)
    interp_enc.runSession(sess_enc)
    
    # 获取 Latent (latent_c)
    out_tensor = interp_enc.getSessionOutput(sess_enc, "output")
    latent_c = np.array(out_tensor.getData(), copy=True).astype(np.float32) 
    
    # 4. Flow Loop
    print("[2/3] Flow Loop (20 steps)...")
    steps = 20
    dt = 1.0 / steps
    
    # 【关键修正】inf.py: x_t = latent_c.clone()
    # 之前 C++ 的噪点就是因为这里搞错了！
    x_t = latent_c.copy()
    
    # 准备输入指针
    t_xt = interp_flow.getSessionInput(sess_flow, "x_t")
    t_xc = interp_flow.getSessionInput(sess_flow, "x_cond")
    t_t  = interp_flow.getSessionInput(sess_flow, "t")
    t_s  = interp_flow.getSessionInput(sess_flow, "s")
    
    # 固定输入: x_cond = latent_c, style = 0
    tmp_xc = MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, latent_c, MNN.Tensor_DimensionType_Caffe)
    t_xc.copyFrom(tmp_xc)
    tmp_s = MNN.Tensor((1,), MNN.Halide_Type_Int, np.array([0], dtype=np.int32), MNN.Tensor_DimensionType_Caffe)
    t_s.copyFrom(tmp_s)
    
    for i in range(steps):
        # 构造 t (标量 float)
        t_val = np.array([i * dt], dtype=np.float32)
        tmp_t = MNN.Tensor((1,), MNN.Halide_Type_Float, t_val, MNN.Tensor_DimensionType_Caffe)
        t_t.copyFrom(tmp_t)
        
        # 更新 x_t
        tmp_xt = MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, x_t, MNN.Tensor_DimensionType_Caffe)
        t_xt.copyFrom(tmp_xt)
        
        interp_flow.runSession(sess_flow)
        
        # 【修复点】获取 velocity (v) 并强制 reshape 回正确形状
        out = interp_flow.getSessionOutput(sess_flow, "output")
        v_flat = np.array(out.getData(), copy=True)
        try:
            v = v_flat.reshape(x_t.shape)   # 优先按 x_t 的 shape 恢复
        except Exception:
            # 回退到预期 shape，保证不会抛错
            v = v_flat.reshape((1, 4, 64, 64))
        # 诊断信息，便于发现 NaN / 非常大值
        print(f"step {i}: v.shape={v.shape} min={v.min():.6f} max={v.max():.6f} has_nan={np.isnan(v).any()}")
        
        # 现在形状对齐了：(1,4,64,64) + (1,4,64,64)
        x_t = x_t + v * dt

    # 5. Decoder
    print("[3/3] Decoder...")
    dec_in = interp_dec.getSessionInput(sess_dec, "input")
    tmp_dec = MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, x_t.astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    dec_in.copyFrom(tmp_dec)
    interp_dec.runSession(sess_dec)
    
    dec_out = interp_dec.getSessionOutput(sess_dec, "output")
    res_flat = np.array(dec_out.getData(), copy=True)
    try:
        res = res_flat.reshape((1, 3, 512, 512))
    except Exception:
        res = res_flat.reshape((1, 3, 512, 512))
    print(f"Decoder out: shape={res.shape} min={res.min():.6f} max={res.max():.6f} has_nan={np.isnan(res).any()}")
    
    # 6. 后处理
    # NCHW -> NHWC
    res = res.transpose(0, 2, 3, 1)[0]
    
    print(f"Result Stats: Min={res.min():.4f}, Max={res.max():.4f}")
    if np.isnan(res).any():
        print("❌ 失败：结果包含 NaN (FP16 溢出或模型损坏)")
    else:
        # 我们在 DecoderWrapper 里已经做了归一化到 0~1，这里直接 *255
        res = np.clip(res, 0.0, 1.0)
        res = (res * 255.0).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imwrite("final_mnn_result.png", res)
        print("✅ 成功！请查看 final_mnn_result.png")

if __name__ == "__main__":
    run()